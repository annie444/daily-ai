# Codebase Optimization Plan

## Goals
1.  **Complexity Reduction:** Remove manual parsing loops, simplify logic.
2.  **Dependency Reduction:** Offload heavy ML to API where appropriate, remove heavy ORM.
3.  **Compositional Design:** Introduce generic Agents and modular Pipelines.
4.  **Local-First Capability:** Ensure the app remains runnable offline / zero-cost via local models.

## Phase 1: Robust AI & Dynamic Prompts (Completed)
*   **Step 1.1: Refactor `ResponseCleaner`:** Updated `Query::from_str` to prioritize `serde_json` and fallback to `ResponseCleaner`.
*   **Step 1.2: Implement `PromptRegistry`:** Created `src/ai/prompt.rs` with `PromptTemplate` struct.
*   **Step 1.3: Update Prompts:** Updated all markdown prompts to use `{{duration}}` variable.

## Phase 2: Core Abstractions (Agentic Design) (Pending)
*   **Step 2.1: `Agent` Struct:** Create `src/ai/agent.rs` with `Agent { client, tools, system_prompt }`. Implement generic `run_task`.
*   **Step 2.2: Standardize Tools:** Ensure `src/ai/tools/` implement a common `Tool` trait.
*   **Step 2.3: Refactor `generate_summary`:** Simplify `summary.rs` to use `Agent`.

## Phase 3: Classification Pipeline Modularization (Completed)
*   **Step 3.1: Traits:** Define `Embedder` and `Clusterer` traits in `src/classify/traits.rs`.
*   **Step 3.2: Modularize:** Refactor `src/classify/mod.rs` to use these traits.

## Phase 4: Workspace & Local ML Isolation (In Progress)
*   **Step 4.1: Workspace:** Convert project to Cargo Workspace. (Completed)
*   **Step 4.2: `local-embedder` Crate:** Extract `src/classify/bert.rs` and heavy dependencies (candle, tokenizers) to `crates/local-embedder`. (Completed)
*   **Step 4.3: `OAIEmbedder`:** Implement API-based embedding. (Completed)
*   **Step 4.4: Feature Gate:** Use `local-ml` feature to toggle the heavy local dependency.
    *   *Status:* `Cargo.toml` updated. `src/classify/bert.rs` currently lacks proper `#[cfg]` guards, causing potential build issues when `local-ml` is disabled. Needs verification and fixing.

## Phase 5: Database Optimization (Pending)
*   **Step 5.1: `sqlx` Replacement:** Replace `sea-orm` with `sqlx` in `src/safari.rs` to reduce binary size (since `sqlx` is already in the tree via `atuin`).
*   **Step 5.2: Dependency Cleanup:** Remove `sea-orm` from `Cargo.toml`.
