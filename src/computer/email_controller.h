#pragma once
// =============================================================================
// Cardinal - Email Controller
// File: src/computer/email_controller.h
//
// IMAP/SMTP email access via Python subprocess (imaplib + smtplib).
// Optional Gmail REST API mode via google-auth + googleapiclient.
//
// Mode is determined by config.computer_use.email.mode:
//   "imap_smtp"  — universal IMAP read + SMTP send
//   "gmail_api"  — Gmail REST API (requires credentials JSON)
// =============================================================================

#include "computer/computer_types.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>

namespace cardinal {

    class EmailController {
    public:
        explicit EmailController(const CardinalConfig& config);
        ~EmailController() = default;

        EmailController(const EmailController&)            = delete;
        EmailController& operator=(const EmailController&) = delete;

        // Check if email is configured and enabled
        bool is_enabled() const;

        // Read emails matching the query
        std::vector<EmailMessage> read(const EmailQuery& query);

        // Send an email
        bool send(const EmailSendRequest& req);

        // Mark email as read
        bool mark_read(const std::string& message_id);

        // Get a specific email by message_id
        std::optional<EmailMessage> get(const std::string& message_id);

    private:
        // Runs a Python script snippet and returns its stdout
        std::string run_python(const std::string& script) const;

        std::string build_imap_read_script(const EmailQuery& q) const;
        std::string build_smtp_send_script(const EmailSendRequest& req) const;
        std::string build_gmail_read_script(const EmailQuery& q) const;
        std::string build_gmail_send_script(const EmailSendRequest& req) const;

        static std::vector<EmailMessage> parse_email_json(const std::string& json_str);
        static std::string python_escape(const std::string& s);

        const CardinalConfig& config_;
    };

} // namespace cardinal
