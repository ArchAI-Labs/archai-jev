//! The `chatml-letters` v1 template: the prompt the default model was trained on (spec 006c).
//!
//! ```text
//! <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\nOption:
//! ```
//!
//! Each question is **one row** (the whole prompt): the state comes after the instructions, so
//! there is no shared prefix. `<|im_start|>` and `<|im_end|>` are ids. The text between them is
//! tokenized as one piece with the fixed text, exactly as the tokenizer does on the training
//! string (added tokens split the text before the pre-tokenizer), and the user's text is sanitized
//! first (the same function as `kev`). The prompt ends at `Option:` with **no space**; the answer
//! is read at that last token.

use super::limits::RequestLimits;
use super::prompt::{Branch, Prompt};
use super::sanitize::sanitize;
use super::tasks::{self, TaskSpec};
use super::template::Template;
use super::tokenize::PromptTokenizer;
use super::unsupported::Unsupported;
use crate::error::{Error, Result};
use crate::schema::{Questions, State, StateValue};

const ROLES: [&str; 2] = ["im_start", "im_end"];

/// The `chatml-letters` template, version 1, with the tasks of one model.
pub struct ChatMlLetters {
    tasks: Option<Vec<TaskSpec>>,
}

/// The instance the template table hands out (no tasks: only for the consistency checks).
pub static CHATML_LETTERS_V1: ChatMlLetters = ChatMlLetters { tasks: None };

impl ChatMlLetters {
    /// The template of a model with a closed set of tasks (`Some`) or, for a model that declares
    /// none, the card's generic format for a `Choice` (`None`).
    pub fn new(tasks: Option<Vec<TaskSpec>>) -> Self {
        ChatMlLetters { tasks }
    }
}

const LETTERS: &str = "ABCDEFGH";

impl Template for ChatMlLetters {
    fn id(&self) -> &'static str {
        "chatml-letters"
    }

    fn version(&self) -> u64 {
        1
    }

    fn roles(&self) -> &'static [&'static str] {
        &ROLES
    }

    fn build(
        &self,
        state: &State,
        questions: &Questions,
        tokenizer: &PromptTokenizer,
        limits: &RequestLimits<'_>,
    ) -> Result<Prompt> {
        let [im_start, im_end] = tokenizer.roles().require(ROLES)?;
        let mut branches = Vec::with_capacity(questions.len());
        for (name, question) in questions.iter() {
            let (system, user) = match &self.tasks {
                Some(specs) => {
                    let task = tasks::find(specs, limits.model, name, question, state)
                        .map_err(Error::Unsupported)?;
                    limits.check_question(name, question)?;
                    let state_text = match state.value() {
                        StateValue::Str(s) => s.as_str(),
                        _ => "",
                    };
                    (
                        task.system.clone(),
                        task.layout.render(
                            &sanitize(state_text),
                            &sanitize(question.instructions().rendered()),
                        ),
                    )
                }
                None => {
                    limits.check_question(name, question)?;
                    generic(name, question, state, limits)?
                }
            };
            let mut ids = vec![im_start];
            ids.extend(tokenizer.encode_raw(
                &format!("system\n{system}"),
                &format!("system prompt of question \"{name}\""),
            )?);
            ids.push(im_end);
            ids.extend(tokenizer.encode_raw("\n", "newline")?);
            ids.push(im_start);
            ids.extend(tokenizer.encode_raw(
                &format!("user\n{user}"),
                &format!("text of question \"{name}\""),
            )?);
            ids.push(im_end);
            ids.extend(tokenizer.encode_raw("\n", "newline")?);
            ids.push(im_start);
            ids.extend(tokenizer.encode_raw("assistant\nOption:", "answer prefix")?);
            limits.check_row(name, 0, ids.len())?;
            let decide = ids.len() - 1;
            branches.push(Branch::new(ids, decide, Vec::new()));
        }
        Ok(Prompt::new(Vec::new(), branches))
    }
}

/// The card's format for a `Choice` of a model that declares no tasks: instructions, then
/// `Option A: ...` lines, then the state. Only choices: the boolean format is not documented.
fn generic(
    name: &str,
    question: &crate::schema::Question,
    state: &State,
    limits: &RequestLimits<'_>,
) -> Result<(String, String)> {
    if !matches!(question, crate::schema::Question::Choice(_)) {
        return Err(Error::Unsupported(Unsupported::QuestionType {
            question: name.to_string(),
            kind: super::limits::kind_name(question).to_string(),
            model: limits.model.to_string(),
            supported: vec!["choice".to_string()],
        }));
    }
    let mut system = sanitize(question.instructions().rendered()).into_owned();
    for (j, text) in question.option_texts().iter().enumerate() {
        let letter = LETTERS.chars().nth(j).unwrap_or('?');
        system.push_str(&format!("\nOption {letter}: {}", sanitize(text)));
    }
    let user = format!("State: {}\nDecision:", sanitize(state.canonical_text()));
    Ok((system, user))
}
