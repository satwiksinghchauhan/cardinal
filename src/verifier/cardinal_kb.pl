% SPDX-License-Identifier: AGPL-3.0-only
% SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
% =============================================================================
% Cardinal Knowledge Base
% File: src/verifier/cardinal_kb.pl
% Core Prolog rules for Cardinal's symbolic verification layer.
% Loaded by SymbolicEngine::init() at startup.
%
% Dynamic predicates (asserted at runtime from rules.json):
%   cardinal_rule(+Id, +Domain, +Condition, +Consequence, +Confidence)
%
% Static predicates (defined here):
%   cardinal_contradicts/6  — contradiction detection
%   cardinal_supports/3     — claim support checking
%   cardinal_all_contradictions/1 — full contradiction scan
%   oppose/2                — domain-specific opposition relations
% =============================================================================

:- module(cardinal, [
    cardinal_rule/5,
    cardinal_contradicts/6,
    cardinal_supports/3,
    cardinal_all_contradictions/1,
    cardinal_query_domain/6
]).

% -----------------------------------------------------------------------------
% Dynamic declarations
% cardinal_rule(+Id, +Domain, +Condition, +Consequence, +Confidence)
% Asserted at runtime by SymbolicEngine::assert_rule()
% -----------------------------------------------------------------------------
:- dynamic cardinal_rule/5.

% -----------------------------------------------------------------------------
% cardinal_contradicts(+Domain, +Condition, +Consequence, -IdA, -IdB, -Explanation)
% Detects if a proposed rule (Domain, Condition, Consequence) contradicts
% an existing rule in the knowledge base.
%
% Contradiction types:
%   1. Direct negation — same condition, opposing consequences
%   2. Domain inconsistency — same domain, mutually exclusive facts
%   3. Causal conflict — A causes B AND A causes not-B
% -----------------------------------------------------------------------------
cardinal_contradicts(Domain, Condition, Consequence, IdA, IdB, Explanation) :-
    cardinal_rule(IdA, Domain, Condition, ExistingConsequence, _),
    ExistingConsequence \= Consequence,
    oppose_consequences(Consequence, ExistingConsequence),
    IdB = 'proposed',
    atomic_list_concat(
        ['Rule ', IdA, ' states that "', Condition,
         '" leads to "', ExistingConsequence,
         '" which opposes proposed consequence "', Consequence, '"'],
        Explanation
    ).

cardinal_contradicts(Domain, _Condition, _Consequence, IdA, IdB, Explanation) :-
    cardinal_rule(IdA, Domain, CondA, ConsequenceA, _),
    cardinal_rule(IdB, Domain, CondB, ConsequenceB, _),
    IdA \= IdB,
    similar_conditions(CondA, CondB),
    oppose_consequences(ConsequenceA, ConsequenceB),
    atomic_list_concat(
        ['Rules ', IdA, ' and ', IdB, ' have similar conditions but opposing consequences'],
        Explanation
    ).

% -----------------------------------------------------------------------------
% cardinal_all_contradictions(-Pairs)
% Returns all contradiction pairs in the current rule base as a list of
% contradiction(IdA, IdB, Explanation) terms.
% -----------------------------------------------------------------------------
cardinal_all_contradictions(Pairs) :-
    findall(
        contradiction(IdA, IdB, Explanation),
        (
            cardinal_rule(IdA, Domain, CondA, ConsA, _),
            cardinal_rule(IdB, Domain, CondB, ConsB, _),
            IdA @< IdB,  % Avoid duplicates
            similar_conditions(CondA, CondB),
            oppose_consequences(ConsA, ConsB),
            atomic_list_concat(
                ['Rules ', IdA, ' and ', IdB,
                 ' conflict: "', CondA, '" leads to both "',
                 ConsA, '" and "', ConsB, '"'],
                Explanation
            )
        ),
        Pairs
    ).

% -----------------------------------------------------------------------------
% cardinal_supports(+Domain, +Claim, -Confidence)
% Checks if the rule base supports a given claim.
% Returns the highest confidence supporting rule's score.
% -----------------------------------------------------------------------------
cardinal_supports(Domain, Claim, Confidence) :-
    cardinal_rule(_, Domain, Condition, Consequence, Confidence),
    (
        sub_atom(Condition,  _, _, _, Claim) ;
        sub_atom(Consequence, _, _, _, Claim)
    ),
    !.  % Cut — return highest confidence (rules are ordered by assertz)

% -----------------------------------------------------------------------------
% cardinal_query_domain(+Domain, +MinConfidence, -Id, -Condition, -Consequence, -Confidence)
% Query all rules for a domain above a confidence threshold.
% -----------------------------------------------------------------------------
cardinal_query_domain(Domain, MinConfidence, Id, Condition, Consequence, Confidence) :-
    cardinal_rule(Id, Domain, Condition, Consequence, Confidence),
    Confidence >= MinConfidence.

% -----------------------------------------------------------------------------
% oppose_consequences(+ConsA, +ConsB)
% True if ConsA and ConsB are logically opposing.
% Uses keyword-based heuristics since rules are natural language.
% -----------------------------------------------------------------------------
oppose_consequences(A, B) :-
    negation_prefix(Prefix),
    atom_concat(Prefix, Rest, A),
    atom_concat(_, Rest, B),
    !.

oppose_consequences(A, B) :-
    negation_prefix(Prefix),
    atom_concat(Prefix, Rest, B),
    atom_concat(_, Rest, A),
    !.

oppose_consequences(A, B) :-
    opposing_pair(A, B), !.

oppose_consequences(A, B) :-
    opposing_pair(B, A), !.

% Negation prefixes
negation_prefix('not ').
negation_prefix('no ').
negation_prefix('never ').
negation_prefix('cannot ').
negation_prefix('does not ').
negation_prefix('is not ').
negation_prefix('are not ').

% Known opposing pairs (extend as Cardinal learns)
opposing_pair('true',  'false').
opposing_pair('yes',   'no').
opposing_pair('valid', 'invalid').
opposing_pair('correct', 'incorrect').
opposing_pair('increases', 'decreases').
opposing_pair('causes',    'prevents').
opposing_pair('enables',   'disables').

% -----------------------------------------------------------------------------
% similar_conditions(+CondA, +CondB)
% True if two conditions are similar enough to potentially conflict.
% Uses substring matching as a heuristic.
% -----------------------------------------------------------------------------
similar_conditions(A, B) :-
    A \= B,
    atom_length(A, LA),
    atom_length(B, LB),
    LA > 3,
    LB > 3,
    (
        sub_atom(A, _, _, _, B) ;
        sub_atom(B, _, _, _, A)
    ).

% -----------------------------------------------------------------------------
% Utility predicates
% -----------------------------------------------------------------------------

% cardinal_rule_count(-N): count asserted rules
cardinal_rule_count(N) :-
    aggregate_all(count, cardinal_rule(_, _, _, _, _), N).

% cardinal_rules_for_domain(+Domain, -Rules): get all rules for a domain
cardinal_rules_for_domain(Domain, Rules) :-
    findall(
        rule(Id, Condition, Consequence, Confidence),
        cardinal_rule(Id, Domain, Condition, Consequence, Confidence),
        Rules
    ).

% cardinal_highest_confidence(+Domain, -Id, -Condition, -Consequence, -Confidence)
% Get the highest confidence rule for a domain
cardinal_highest_confidence(Domain, Id, Condition, Consequence, MaxConf) :-
    findall(
        Conf-rule(I, Cond, Cons),
        cardinal_rule(I, Domain, Cond, Cons, Conf),
        Pairs
    ),
    Pairs \= [],
    max_member(MaxConf-rule(Id, Condition, Consequence), Pairs).