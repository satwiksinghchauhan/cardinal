// =============================================================================
// Cardinal - Symbolic Engine Implementation
// File: src/verifier/symbolic_engine.cpp
// =============================================================================

#include "symbolic_engine.h"
#include "utils/logger.h"

#include <sstream>
#include <filesystem>
#include <cstring>

namespace cardinal {

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    SymbolicEngine::SymbolicEngine(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("SymbolicEngine created Ã¢â‚¬â€ mode: " + config_.verifier.mode);
    }

    SymbolicEngine::~SymbolicEngine() {
        shutdown();
    }

    // =============================================================================
    // init
    // =============================================================================

    void SymbolicEngine::init(const std::string& kb_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) {
            LOG_WARN("SymbolicEngine: already initialized");
            return;
        }

        // Determine KB path
        std::string kb = kb_path.empty()
            ? "src/verifier/cardinal_kb.pl"
            : kb_path;

        if (!std::filesystem::exists(kb)) {
            throw SymbolicEngineError("Knowledge base not found: " + kb);
        }

        LOG_INFO("Initializing SWI-Prolog engine...");
        LOG_INFO("Knowledge base: " + kb);

        // Build argv for PL_initialise
        // -x: no startup goal
        // -f: load file on startup
        // -g: run goal after load
        // --quiet: suppress Prolog startup messages
        pl_argv_strings_.clear();
        pl_argv_strings_.push_back("cardinal");
        pl_argv_strings_.push_back("--quiet");
        pl_argv_strings_.push_back("-g");
        pl_argv_strings_.push_back("true");
        pl_argv_strings_.push_back("-f");
        pl_argv_strings_.push_back(kb);

        pl_argv_.clear();
        for (auto& s : pl_argv_strings_) {
            pl_argv_.push_back(const_cast<char*>(s.c_str()));
        }
        pl_argv_.push_back(nullptr);

        int argc = static_cast<int>(pl_argv_.size()) - 1;

        if (!PL_initialise(argc, pl_argv_.data())) {
            throw SymbolicEngineError("PL_initialise failed Ã¢â‚¬â€ SWI-Prolog could not start");
        }

        initialized_ = true;

        // Verify KB loaded correctly by checking if cardinal_rule/5 is defined
        if (!query_once("current_predicate(cardinal_rule/5)")) {
            // Define it dynamically if not already declared in KB
            query_once(":- dynamic cardinal_rule/5");
            LOG_DEBUG("SymbolicEngine: declared cardinal_rule/5 dynamically");
        }

        LOG_INFO("SWI-Prolog engine initialized successfully");
    }

    // =============================================================================
    // shutdown
    // =============================================================================

    void SymbolicEngine::shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_) return;

        PL_halt(0);
        initialized_ = false;
        asserted_rules_ = 0;

        LOG_INFO("SymbolicEngine shut down");
    }

    // =============================================================================
    // assert_rule
    // =============================================================================

    RuleAssertResult SymbolicEngine::assert_rule(const Rule& rule) {
        std::lock_guard<std::mutex> lock(mutex_);

        RuleAssertResult result;
        result.success = false;

        if (!initialized_) {
            result.error_message = "Engine not initialized";
            return result;
        }

        std::string fact = rule_to_prolog_fact(rule);
        std::string goal = "assertz(" + fact + ")";

        result.prolog_fact = fact;

        fid_t fid = PL_open_foreign_frame();

        term_t goal_term = PL_new_term_ref();
        if (!PL_chars_to_term(goal.c_str(), goal_term)) {
            result.error_message = "Failed to parse goal: " + goal;
            PL_close_foreign_frame(fid);
            return result;
        }

        int rc = PL_call(goal_term, nullptr);
        if (rc) {
            result.success = true;
            ++asserted_rules_;
            LOG_DEBUG("SymbolicEngine: asserted rule [" + rule.domain + "] " + rule.id);
        }
        else {
            result.error_message = get_prolog_error();
            LOG_WARN("SymbolicEngine: failed to assert rule " + rule.id +
                ": " + result.error_message);
        }

        PL_close_foreign_frame(fid);
        return result;
    }

    // =============================================================================
    // assert_rules (bulk)
    // =============================================================================

    int SymbolicEngine::assert_rules(const std::vector<Rule>& rules) {
        int succeeded = 0;
        for (const auto& rule : rules) {
            auto result = assert_rule(rule);
            if (result.success) ++succeeded;
        }
        LOG_INFO("SymbolicEngine: asserted " + std::to_string(succeeded) +
            "/" + std::to_string(rules.size()) + " rules");
        return succeeded;
    }

    // =============================================================================
    // retract_rule
    // =============================================================================

    bool SymbolicEngine::retract_rule(const std::string& rule_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return false;

        std::string goal = "retract(cardinal_rule('" +
            escape_prolog_atom(rule_id) +
            "', _, _, _, _))";

        fid_t fid = PL_open_foreign_frame();
        term_t t = PL_new_term_ref();
        bool ok = false;

        if (PL_chars_to_term(goal.c_str(), t)) {
            ok = PL_call(t, nullptr) != 0;
            if (ok) --asserted_rules_;
        }

        PL_close_foreign_frame(fid);
        return ok;
    }

    // =============================================================================
    // retract_all_rules
    // =============================================================================

    void SymbolicEngine::retract_all_rules() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;

        fid_t fid = PL_open_foreign_frame();
        term_t t = PL_new_term_ref();

        if (PL_chars_to_term("retractall(cardinal_rule(_, _, _, _, _))", t)) {
            PL_call(t, nullptr);
        }

        asserted_rules_ = 0;
        PL_close_foreign_frame(fid);
        LOG_DEBUG("SymbolicEngine: retracted all rules");
    }

    // =============================================================================
    // check_contradiction
    // Runs the Prolog contradiction predicate against a new rule candidate.
    // The KB's contradicts/4 predicate handles the logic.
    // =============================================================================

    ContradictionResult SymbolicEngine::check_contradiction(
        const std::string& domain,
        const std::string& condition,
        const std::string& consequence) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        ContradictionResult result;
        result.has_contradiction = false;
        result.severity = 0.0f;

        if (!initialized_) return result;

        // Use predicate-based query (more reliable than string parsing)
        fid_t fid = PL_open_foreign_frame();

        term_t args = PL_new_term_refs(6);
        PL_put_atom_chars(args + 0, domain.c_str());
        PL_put_atom_chars(args + 1, condition.c_str());
        PL_put_atom_chars(args + 2, consequence.c_str());
        PL_put_variable(args + 3);  // IdA
        PL_put_variable(args + 4);  // IdB
        PL_put_variable(args + 5);  // Explanation

        predicate_t pred = PL_predicate("cardinal_contradicts", 6, "user");
        if (!pred) {
            PL_close_foreign_frame(fid);
            return result;
        }

        qid_t qid = PL_open_query(nullptr, 0, pred, args);
        if (!qid) {
            PL_close_foreign_frame(fid);
            return result;
        }

        if (PL_next_solution(qid)) {
            result.has_contradiction = true;
            result.severity = 0.8f;

            char* id_a_str = nullptr;
            char* id_b_str = nullptr;
            char* expl_str = nullptr;

            if (PL_get_atom_chars(args + 3, &id_a_str))
                result.rule_id_a = id_a_str ? id_a_str : "";
            if (PL_get_atom_chars(args + 4, &id_b_str))
                result.rule_id_b = id_b_str ? id_b_str : "";
            if (PL_get_atom_chars(args + 5, &expl_str))
                result.explanation = expl_str ? expl_str : "contradiction detected";

            LOG_WARN("SymbolicEngine: contradiction detected between '" +
                result.rule_id_a + "' and '" + result.rule_id_b + "'");
        }

        PL_close_query(qid);
        PL_close_foreign_frame(fid);

        return result;
    }

    // =============================================================================
    // check_all_contradictions
    // =============================================================================

    std::vector<ContradictionResult> SymbolicEngine::check_all_contradictions() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<ContradictionResult> results;
        if (!initialized_) return results;

        // Query all contradicting pairs in the rule base
        std::string goal = "cardinal_all_contradictions(Pairs)";

        fid_t fid = PL_open_foreign_frame();
        term_t t = PL_new_term_ref();

        if (PL_chars_to_term(goal.c_str(), t)) {
            term_t pairs = PL_new_term_ref();
            predicate_t pred = PL_predicate("cardinal_all_contradictions", 1, "user");
            qid_t qid = PL_open_query(nullptr, PL_Q_CATCH_EXCEPTION, pred, pairs);

            if (qid && PL_next_solution(qid)) {
                // Parse the list of contradiction pairs
                term_t head = PL_new_term_ref();
                term_t list = PL_copy_term_ref(pairs);

                while (PL_get_list(list, head, list)) {
                    ContradictionResult cr;
                    cr.has_contradiction = true;
                    cr.severity = 0.8f;

                    // Each element is a pair(IdA, IdB, Explanation)
                    term_t a = PL_new_term_ref();
                    term_t b = PL_new_term_ref();
                    term_t expl = PL_new_term_ref();

                    if (PL_get_arg(1, head, a) &&
                        PL_get_arg(2, head, b) &&
                        PL_get_arg(3, head, expl)) {
                        char* sa = nullptr;
                        char* sb = nullptr;
                        char* se = nullptr;
                        if (PL_get_atom_chars(a, &sa))    cr.rule_id_a = sa ? sa : "";
                        if (PL_get_atom_chars(b, &sb))    cr.rule_id_b = sb ? sb : "";
                        if (PL_get_atom_chars(expl, &se)) cr.explanation = se ? se : "";
                    }

                    results.push_back(cr);
                }
            }

            if (qid) PL_close_query(qid);
        }

        PL_close_foreign_frame(fid);

        if (!results.empty()) {
            LOG_WARN("SymbolicEngine: found " + std::to_string(results.size()) +
                " contradiction(s) in rule base");
        }

        return results;
    }

    // =============================================================================
    // query_rules
    // =============================================================================

    std::vector<Rule> SymbolicEngine::query_rules(
        const std::string& domain,
        float min_confidence,
        int   max_results) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Rule> results;
        if (!initialized_) return results;

        fid_t fid = PL_open_foreign_frame();

        term_t args = PL_new_term_refs(5);
        PL_put_atom_chars(args + 0, "");       // Id (unbound)
        PL_put_atom_chars(args + 1, domain.c_str()); // Domain
        PL_put_atom_chars(args + 2, "");       // Condition (unbound)
        PL_put_atom_chars(args + 3, "");       // Consequence (unbound)
        PL_put_float(args + 4, 0.0);           // Confidence (unbound)

        // Use variables for unbound args
        PL_put_variable(args + 0);
        PL_put_variable(args + 2);
        PL_put_variable(args + 3);
        PL_put_variable(args + 4);

        predicate_t pred = PL_predicate("cardinal_rule", 5, "user");
        qid_t qid = PL_open_query(nullptr, PL_Q_CATCH_EXCEPTION, pred, args);

        if (qid) {
            int count = 0;
            while (PL_next_solution(qid) &&
                (max_results <= 0 || count < max_results)) {
                Rule rule;

                char* id_str = nullptr;
                char* cond_str = nullptr;
                char* cons_str = nullptr;
                double conf = 0.0;

                if (PL_get_atom_chars(args + 0, &id_str))
                    rule.id = id_str ? id_str : "";
                rule.domain = domain;
                if (PL_get_atom_chars(args + 2, &cond_str))
                    rule.condition = cond_str ? cond_str : "";
                if (PL_get_atom_chars(args + 3, &cons_str))
                    rule.consequence = cons_str ? cons_str : "";
                if (PL_get_float(args + 4, &conf))
                    rule.confidence = static_cast<float>(conf);

                if (rule.confidence >= min_confidence && !rule.id.empty()) {
                    results.push_back(rule);
                    ++count;
                }
            }
            PL_close_query(qid);
        }

        PL_close_foreign_frame(fid);
        return results;
    }

    // =============================================================================
    // check_claim
    // =============================================================================

    float SymbolicEngine::check_claim(const std::string& domain,
        const std::string& claim) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return 0.0f;

        // Query: cardinal_supports(Domain, Claim, Confidence)
        fid_t fid = PL_open_foreign_frame();

        term_t args = PL_new_term_refs(3);
        PL_put_atom_chars(args + 0, domain.c_str());
        PL_put_atom_chars(args + 1, claim.c_str());
        PL_put_variable(args + 2);

        float confidence = 0.0f;

        predicate_t pred = PL_predicate("cardinal_supports", 3, "user");
        qid_t qid = PL_open_query(nullptr, PL_Q_CATCH_EXCEPTION, pred, args);

        if (qid && PL_next_solution(qid)) {
            double conf = 0.0;
            if (PL_get_float(args + 2, &conf)) {
                confidence = static_cast<float>(conf);
            }
            PL_close_query(qid);
        }
        else if (qid) {
            PL_close_query(qid);
        }

        PL_close_foreign_frame(fid);
        return confidence;
    }

    // =============================================================================
    // query / query_once
    // =============================================================================

    PrologResult SymbolicEngine::query(const std::string& goal) const {
        std::lock_guard<std::mutex> lock(mutex_);

        PrologResult result;
        result.success = false;
        result.deterministic = false;

        if (!initialized_) {
            result.error_message = "Engine not initialized";
            return result;
        }

        fid_t fid = PL_open_foreign_frame();

        term_t goal_term = PL_new_term_ref();
        if (!PL_chars_to_term(goal.c_str(), goal_term)) {
            result.error_message = "Failed to parse goal: " + goal;
            PL_close_foreign_frame(fid);
            return result;
        }

        qid_t qid = PL_open_query(nullptr,
            PL_Q_CATCH_EXCEPTION | PL_Q_NODEBUG,
            nullptr, goal_term);
        if (!qid) {
            result.error_message = "Failed to open query";
            PL_close_foreign_frame(fid);
            return result;
        }

        int rc = PL_next_solution(qid);
        if (rc == TRUE) {
            result.success = true;
        }
        else if (rc == FALSE) {
            result.success = false;
        }
        else {
            // Exception
            term_t ex = PL_exception(qid);
            if (ex) {
                result.error_message = get_prolog_error();
            }
        }

        PL_close_query(qid);
        PL_close_foreign_frame(fid);
        return result;
    }

    bool SymbolicEngine::query_once(const std::string& goal) const {
        // Note: this version doesn't lock since it's called from init()
        // which already holds the lock, and from query() which also locks.
        // We use a non-locking implementation here.

        fid_t fid = PL_open_foreign_frame();

        term_t t = PL_new_term_ref();
        bool ok = false;

        if (PL_chars_to_term(goal.c_str(), t)) {
            ok = PL_call(t, nullptr) != 0;
        }

        PL_close_foreign_frame(fid);
        return ok;
    }

    // =============================================================================
    // Internal helpers
    // =============================================================================

    std::string SymbolicEngine::escape_prolog_atom(const std::string& s) const {
        std::string result;
        result.reserve(s.size());
        for (char c : s) {
            if (c == '\'') result += "\\'";
            else if (c == '\\') result += "\\\\";
            else result += c;
        }
        return result;
    }

    std::string SymbolicEngine::rule_to_prolog_fact(const Rule& rule) const {
        std::ostringstream oss;
        oss << "cardinal_rule("
            << "'" << escape_prolog_atom(rule.id) << "',"
            << "'" << escape_prolog_atom(rule.domain) << "',"
            << "'" << escape_prolog_atom(rule.condition) << "',"
            << "'" << escape_prolog_atom(rule.consequence) << "',"
            << rule.confidence
            << ")";
        return oss.str();
    }

    std::string SymbolicEngine::term_to_string(term_t t) const {
        char* s = nullptr;
        if (PL_get_atom_chars(t, &s) && s) return s;

        // Try as string
        size_t len = 0;
        char* buf = nullptr;
        if (PL_get_string_chars(t, &buf, &len) && buf) return std::string(buf, len);

        // Fallback: write_term
        std::string result;
        fid_t fid = PL_open_foreign_frame();
        term_t str = PL_new_term_ref();
        if (PL_call_predicate(nullptr, PL_Q_NODEBUG,
            PL_predicate("term_string", 2, "system"),
            t)) {
            char* out = nullptr;
            if (PL_get_atom_chars(str, &out) && out) result = out;
        }
        PL_close_foreign_frame(fid);
        return result;
    }

    std::string SymbolicEngine::get_prolog_error() const {
        term_t ex = PL_exception(0);
        if (!ex) return "unknown Prolog error";
        return term_to_string(ex);
    }

} // namespace cardinal