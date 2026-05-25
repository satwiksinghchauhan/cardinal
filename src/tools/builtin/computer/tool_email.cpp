// =============================================================================
// Cardinal - Tool: email Implementation
// =============================================================================

#include "tools/builtin/computer/tool_email.h"
#include "computer/email_controller.h"

#include <sstream>
#include <chrono>

namespace cardinal {

ToolDefinition make_email_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "email";
    def.description =
        "Read or send email via IMAP/SMTP or Gmail API.\n\n"
        "Actions:\n"
        "  read — search and retrieve emails\n"
        "  send — compose and send an email";
    def.confirmation_required = config.computer_use.safety.confirmation_required;

    def.parameters.push_back({
        "action", ToolParameterType::STRING,
        "Action: read|send",
        true, ""
    });
    // Read params
    def.parameters.push_back({
        "folder", ToolParameterType::STRING,
        "Folder/mailbox to read from. Default: INBOX",
        false, "INBOX"
    });
    def.parameters.push_back({
        "subject", ToolParameterType::STRING,
        "Filter by subject containing this string",
        false, ""
    });
    def.parameters.push_back({
        "from", ToolParameterType::STRING,
        "Filter by sender containing this string",
        false, ""
    });
    def.parameters.push_back({
        "unread_only", ToolParameterType::BOOLEAN,
        "If true, return only unread messages. Default: false",
        false, "false"
    });
    def.parameters.push_back({
        "max_results", ToolParameterType::NUMBER,
        "Maximum emails to return. Default: 10",
        false, "10"
    });
    // Send params
    def.parameters.push_back({
        "to", ToolParameterType::STRING,
        "Recipient email address(es), comma-separated",
        false, ""
    });
    def.parameters.push_back({
        "send_subject", ToolParameterType::STRING,
        "Email subject line",
        false, ""
    });
    def.parameters.push_back({
        "body", ToolParameterType::STRING,
        "Email body text",
        false, ""
    });
    return def;
}

ToolResult execute_email(const ToolCall& call, EmailController& email) {
    ToolResult result;
    result.tool_name = "email";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    auto get = [&](const std::string& k, const std::string& d = "") {
        auto it = call.arguments.find(k);
        return it != call.arguments.end() ? it->second : d;
    };

    try {
        if (!email.is_enabled()) {
            result.status = ToolStatus::DISABLED;
            result.output = "Email is not configured. Set computer_use.email.enabled=true in config.json.";
            return result;
        }

        std::string action = get("action");

        if (action == "read") {
            EmailQuery q;
            q.folder           = get("folder", "INBOX");
            q.subject_contains = get("subject");
            q.from_contains    = get("from");
            q.unread_only      = get("unread_only", "false") == "true";
            try { q.max_results = std::stoi(get("max_results", "10")); } catch (...) {}

            auto messages = email.read(q);

            if (messages.empty()) {
                result.status = ToolStatus::SUCCESS;
                result.output = "No emails found matching the criteria.";
            } else {
                std::ostringstream oss;
                oss << "Found " << messages.size() << " email(s):\n\n";
                for (size_t i = 0; i < messages.size(); ++i) {
                    const auto& m = messages[i];
                    oss << "--- Email " << (i+1) << " ---\n"
                        << "From:    " << m.from    << "\n"
                        << "Subject: " << m.subject << "\n"
                        << "Date:    " << m.date    << "\n"
                        << "Body:\n"
                        << m.body_text.substr(0, 1000)
                        << (m.body_text.size() > 1000 ? "\n[truncated]" : "")
                        << "\n\n";
                }
                result.status = ToolStatus::SUCCESS;
                result.output = oss.str();
            }

        } else if (action == "send") {
            EmailSendRequest req;
            // Parse comma-separated recipients
            std::string to_str = get("to");
            std::istringstream ss(to_str);
            std::string addr;
            while (std::getline(ss, addr, ',')) {
                auto& s = addr;
                while (!s.empty() && s.front() == ' ') s.erase(s.begin());
                while (!s.empty() && s.back()  == ' ') s.pop_back();
                if (!s.empty()) req.to.push_back(s);
            }
            req.subject = get("send_subject");
            req.body    = get("body");

            if (req.to.empty() || req.subject.empty() || req.body.empty()) {
                result.status = ToolStatus::INVALID_ARGS;
                result.output = "send requires: to, send_subject, body";
                return result;
            }

            bool ok = email.send(req);
            result.status = ok ? ToolStatus::SUCCESS : ToolStatus::FAILURE;
            result.output = ok
                ? "Email sent to " + to_str
                : "Failed to send email. Check email config and CARDINAL_EMAIL_PASS env var.";

        } else {
            result.status = ToolStatus::INVALID_ARGS;
            result.output = "Unknown action: " + action + ". Use read or send.";
        }
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "email tool failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal
