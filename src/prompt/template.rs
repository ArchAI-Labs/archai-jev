//! Prompt templates (D19): pure, versioned functions from a request to ids.
//!
//! A template is tied to a family. Changing anything that changes one id is a new version with a
//! new golden; the model-loading code checks that the template named by a manifest exists and
//! needs the roles the family says (spec 005 section 6.1, spec 006a section 2).

use super::chatml::{CHATML_LETTERS_V1, ChatMlLetters};
use super::kev::{KEV_V1, KevV1};
use super::limits::RequestLimits;
use super::prompt::Prompt;
use super::tasks;
use super::tokenize::PromptTokenizer;
use crate::error::Result;
use crate::models::incompat::Incompat;
use crate::models::manifest::Task;
use crate::schema::{Questions, State};

/// A prompt template.
pub trait Template: Send + Sync {
    /// Template id as written in manifests (`kev`).
    fn id(&self) -> &'static str;

    /// Template version (`1`).
    fn version(&self) -> u64;

    /// The special-token roles the template needs; no more, no fewer.
    fn roles(&self) -> &'static [&'static str];

    /// Build the prompt for `state` and `questions`, refusing (never truncating) what the model
    /// cannot answer.
    ///
    /// # Errors
    /// [`crate::Error::Unsupported`] for the reasons of spec 006a section 6.
    fn build(
        &self,
        state: &State,
        questions: &Questions,
        tokenizer: &PromptTokenizer,
        limits: &RequestLimits<'_>,
    ) -> Result<Prompt>;
}

/// The template called `id` at `version`, if it exists (without the tasks of any model).
pub fn lookup(id: &str, version: u64) -> Option<&'static dyn Template> {
    let all: [&'static dyn Template; 2] = [&KEV_V1, &CHATML_LETTERS_V1];
    all.into_iter()
        .find(|t| t.id() == id && t.version() == version)
}

/// The template of a model: the one called `id` at `version`, with the tasks of the manifest
/// (only `chatml-letters` reads them).
///
/// # Errors
/// [`Incompat`] if the template does not exist or the tasks are not valid for it.
pub fn instantiate(
    id: &str,
    version: u64,
    tasks: Option<&[Task]>,
    max_options: usize,
) -> std::result::Result<Box<dyn Template>, Incompat> {
    match (id, version) {
        ("kev", 1) => Ok(Box::new(KevV1)),
        ("chatml-letters", 1) => {
            let specs = tasks.map(|t| tasks::parse(t, max_options)).transpose()?;
            Ok(Box::new(ChatMlLetters::new(specs)))
        }
        _ => Err(Incompat::TemplateMismatch {
            family: String::new(),
            expected: "kev v1 or chatml-letters v1".to_string(),
            got: format!("{id} v{version}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::families;

    /// Families whose template does not exist (the fake family of the loader tests). A family
    /// that is not listed here **must** have its template.
    const PENDING: [&str; 1] = ["test-fake"];

    #[test]
    fn kev_v1_roles() {
        let t = lookup("kev", 1).expect("kev v1");
        assert_eq!(
            t.roles(),
            [
                "fim_prefix",
                "fim_middle",
                "fim_suffix",
                "box_start",
                "box_end"
            ]
        );
        assert!(lookup("kev", 2).is_none());
        assert!(lookup("nope", 1).is_none());
    }

    #[test]
    fn every_family_template_exists_with_the_roles_of_the_family() {
        let mut checked = 0;
        for key in families::keys() {
            let family = families::lookup(&key).unwrap();
            match lookup(family.template_id, family.template_version) {
                Some(t) => {
                    let mut a: Vec<_> = t.roles().to_vec();
                    let mut b: Vec<_> = family.special_roles.to_vec();
                    a.sort_unstable();
                    b.sort_unstable();
                    assert_eq!(a, b, "family {key}");
                    checked += 1;
                }
                None => assert!(
                    PENDING.contains(&key.as_str()),
                    "family {key} declares template {} v{} which does not exist",
                    family.template_id,
                    family.template_version
                ),
            }
        }
        assert!(checked >= 1, "test-kev must be checked");
    }
}
