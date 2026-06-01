// =============================================================================
// Cardinal - Email Controller Implementation
// File: src/computer/email_controller.cpp
// =============================================================================

#include "computer/email_controller.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <array>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

namespace cardinal {

EmailController::EmailController(const CardinalConfig& config)
    : config_(config)
{}

bool EmailController::is_enabled() const {
    return config_.computer_use.email.enabled;
}

// ---------------------------------------------------------------------------
// Python escape helper
// ---------------------------------------------------------------------------

std::string EmailController::python_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '\'') out += "\\'";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    return out;
}

// ---------------------------------------------------------------------------
// run_python
// ---------------------------------------------------------------------------

std::string EmailController::run_python(const std::string& script) const {
    // Write script to temp file and run with system python3
    std::string path = "/tmp/cardinal_email_" + std::to_string(getpid()) + ".py";
    {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) return "";
        fputs(script.c_str(), f);
        fclose(f);
    }

    std::string cmd = "python3 " + path + " 2>/dev/null";
    std::array<char, 8192> buf{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { remove(path.c_str()); return ""; }
    while (fgets(buf.data(), buf.size(), pipe)) output += buf.data();
    pclose(pipe);
    remove(path.c_str());
    return output;
}

// ---------------------------------------------------------------------------
// IMAP read script
// ---------------------------------------------------------------------------

std::string EmailController::build_imap_read_script(const EmailQuery& q) const {
    const auto& ec = config_.computer_use.email;
    std::ostringstream s;
    s << "import imaplib, email, json, os\n"
      << "from email.header import decode_header\n\n"
      << "def decode_str(s):\n"
      << "    if not s: return ''\n"
      << "    parts = decode_header(s)\n"
      << "    result = ''\n"
      << "    for b, enc in parts:\n"
      << "        if isinstance(b, bytes): result += b.decode(enc or 'utf-8', errors='replace')\n"
      << "        else: result += str(b)\n"
      << "    return result\n\n"
      << "try:\n"
      << "    m = imaplib.IMAP4_SSL('" << python_escape(ec.imap_host) << "', "
                                       << ec.imap_port << ")\n"
      << "    m.login('" << python_escape(ec.address) << "', os.environ.get('CARDINAL_EMAIL_PASS',''))\n"
      << "    m.select('" << python_escape(q.folder) << "')\n"
      << "    search_parts = []\n";

    if (q.unread_only) s << "    search_parts.append('UNSEEN')\n";
    if (!q.subject_contains.empty())
        s << "    search_parts.append('SUBJECT \"" << python_escape(q.subject_contains) << "\"')\n";
    if (!q.from_contains.empty())
        s << "    search_parts.append('FROM \"" << python_escape(q.from_contains) << "\"')\n";

    s << "    criteria = ' '.join(search_parts) if search_parts else 'ALL'\n"
      << "    typ, data = m.search(None, criteria)\n"
      << "    ids = data[0].split()\n"
      << "    ids = ids[-" << q.max_results << ":]\n"
      << "    messages = []\n"
      << "    for num in reversed(ids):\n"
      << "        typ, msg_data = m.fetch(num, '(RFC822)')\n"
      << "        msg = email.message_from_bytes(msg_data[0][1])\n"
      << "        body = ''\n"
      << "        if msg.is_multipart():\n"
      << "            for part in msg.walk():\n"
      << "                if part.get_content_type() == 'text/plain':\n"
      << "                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')\n"
      << "                    break\n"
      << "        else:\n"
      << "            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')\n"
      << "        messages.append({\n"
      << "            'message_id': msg.get('Message-ID',''),\n"
      << "            'from': decode_str(msg.get('From','')),\n"
      << "            'subject': decode_str(msg.get('Subject','')),\n"
      << "            'date': msg.get('Date',''),\n"
      << "            'body_text': body[:5000]\n"
      << "        })\n"
      << "    m.logout()\n"
      << "    print(json.dumps(messages))\n"
      << "except Exception as e:\n"
      << "    print(json.dumps({'error': str(e)}))\n";
    return s.str();
}

// ---------------------------------------------------------------------------
// SMTP send script
// ---------------------------------------------------------------------------

std::string EmailController::build_smtp_send_script(const EmailSendRequest& req) const {
    const auto& ec = config_.computer_use.email;
    std::string to_list;
    for (size_t i = 0; i < req.to.size(); ++i) {
        if (i) to_list += ", ";
        to_list += python_escape(req.to[i]);
    }

    std::ostringstream s;
    s << "import smtplib, os\n"
      << "from email.mime.text import MIMEText\n"
      << "from email.mime.multipart import MIMEMultipart\n\n"
      << "try:\n"
      << "    msg = MIMEMultipart()\n"
      << "    msg['From'] = '" << python_escape(ec.address) << "'\n"
      << "    msg['To'] = '" << to_list << "'\n"
      << "    msg['Subject'] = '" << python_escape(req.subject) << "'\n"
      << "    msg.attach(MIMEText('" << python_escape(req.body) << "', '"
                                    << (req.html ? "html" : "plain") << "'))\n"
      << "    with smtplib.SMTP('" << python_escape(ec.smtp_host) << "', "
                                  << ec.smtp_port << ") as server:\n"
      << "        server.starttls()\n"
      << "        server.login('" << python_escape(ec.address)
                                 << "', os.environ.get('CARDINAL_EMAIL_PASS',''))\n"
      << "        server.send_message(msg)\n"
      << "    print('ok')\n"
      << "except Exception as e:\n"
      << "    print('error:' + str(e))\n";
    return s.str();
}

// ---------------------------------------------------------------------------
// parse_email_json
// ---------------------------------------------------------------------------

std::vector<EmailMessage> EmailController::parse_email_json(const std::string& json_str) {
    std::vector<EmailMessage> result;
    if (json_str.empty()) return result;

    try {
        auto j = json::parse(json_str);
        if (j.is_object() && j.contains("error")) return result;
        if (!j.is_array()) return result;
        for (const auto& item : j) {
            EmailMessage m;
            m.message_id = item.value("message_id", "");
            m.from       = item.value("from", "");
            m.subject    = item.value("subject", "");
            m.date       = item.value("date", "");
            m.body_text  = item.value("body_text", "");
            result.push_back(m);
        }
    } catch (...) {}
    return result;
}

// ---------------------------------------------------------------------------
// Public methods
// ---------------------------------------------------------------------------

std::vector<EmailMessage> EmailController::read(const EmailQuery& query) {
    if (!is_enabled()) return {};

    std::string script;
    if (config_.computer_use.email.mode == "gmail_api")
        script = build_gmail_read_script(query);
    else
        script = build_imap_read_script(query);

    std::string output = run_python(script);
    return parse_email_json(output);
}

bool EmailController::send(const EmailSendRequest& req) {
    if (!is_enabled()) return false;

    std::string script;
    if (config_.computer_use.email.mode == "gmail_api")
        script = build_gmail_send_script(req);
    else
        script = build_smtp_send_script(req);

    std::string output = run_python(script);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r'))
        output.pop_back();
    return output == "ok";
}

bool EmailController::mark_read(const std::string& /*message_id*/) {
    // IMAP flag via python — simplified placeholder
    LOG_DEBUG("EmailController: mark_read not yet implemented");
    return false;
}

std::optional<EmailMessage> EmailController::get(const std::string& /*message_id*/) {
    LOG_DEBUG("EmailController: get not yet implemented");
    return std::nullopt;
}

// Gmail API stubs (requires google-auth in venv)
std::string EmailController::build_gmail_read_script(const EmailQuery& q) const {
    (void)q;
    return "print('[]')\n"; // placeholder — full impl via googleapiclient
}

std::string EmailController::build_gmail_send_script(const EmailSendRequest& req) const {
    (void)req;
    return "print('error:gmail_api_not_implemented')\n";
}

} // namespace cardinal
