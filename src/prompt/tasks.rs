//! The closed set of tasks of a letters model (spec 006c sections 1-3).
//!
//! The default model was trained on four fixed prompts and never saw the options changed or
//! permuted, so a question is answered **only** if it is exactly one of those tasks. The manifest
//! carries, for each task, how to recognise the question and the literal system text; this module
//! reads that (strictly), and decides whether a question corresponds to a task.

use crate::json_strict::{Json, Obj};
use crate::models::incompat::Incompat;
use crate::models::manifest::Task;
use crate::prompt::Unsupported;
use crate::schema::{Question, State, StateValue};

/// How the user part of the prompt is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `State: {s}\nDecision:`
    State,
    /// `State: {s}\nCondition: {i}\nDecision:`
    StateCondition,
    /// `State A: {s}\nState B: {i}\nDecision:`
    StatePair,
}

impl Layout {
    fn parse(name: &str) -> Option<Layout> {
        match name {
            "state" => Some(Layout::State),
            "state-condition" => Some(Layout::StateCondition),
            "state-pair" => Some(Layout::StatePair),
            _ => None,
        }
    }

    /// The user text after `user\n`, with `state` and `instructions` already sanitized.
    pub fn render(self, state: &str, instructions: &str) -> String {
        match self {
            Layout::State => format!("State: {state}\nDecision:"),
            Layout::StateCondition => {
                format!("State: {state}\nCondition: {instructions}\nDecision:")
            }
            Layout::StatePair => format!("State A: {state}\nState B: {instructions}\nDecision:"),
        }
    }
}

/// How a question is recognised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// A `Choice` with these instructions and these keys (and, if given, these descriptions).
    Choice {
        /// Exact instructions.
        instructions: String,
        /// `(key, description)` in order.
        options: Vec<(String, Option<String>)>,
    },
    /// A `YesNo` with these two descriptions; the instructions are free (they are the condition).
    YesNo {
        /// Description of the true answer.
        true_description: String,
        /// Description of the false answer.
        false_description: String,
    },
}

/// One declared task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    /// Stable id (`safety`, ...).
    pub id: String,
    /// How to recognise the question.
    pub rule: Rule,
    /// The literal system text of training.
    pub system: String,
    /// How the user part is laid out.
    pub layout: Layout,
}

fn text(item: &Json, path: &str) -> Result<String, Incompat> {
    match item {
        Json::Str(s) => Ok(s.clone()),
        other => Err(Incompat::ManifestBadValue {
            path: path.to_string(),
            detail: format!("must be a string, got {}", other.kind()),
        }),
    }
}

fn bad(path: &str, detail: &str) -> Incompat {
    Incompat::ManifestBadValue {
        path: path.to_string(),
        detail: detail.to_string(),
    }
}

/// Read the `tasks` array of a manifest from a JSON value (the same shape as in the manifest).
///
/// # Errors
/// [`Incompat`] naming the field.
pub fn tasks_from_json(json: &Json) -> Result<Vec<Task>, Incompat> {
    let Json::Array(items) = json else {
        return Err(bad("tasks", "must be an array"));
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let mut t = Obj::new(item, &format!("tasks[{i}]"))?;
        let id = t.str("id")?.to_string();
        let kind = t.str("kind")?.to_string();
        let matching = t.req("match")?.clone();
        t.finish()?;
        out.push(Task { id, kind, matching });
    }
    Ok(out)
}

/// The four tasks of the default model, as declared in its manifest (spec 006c section 1).
///
/// # Panics
/// Never in practice: the JSON is part of the source and covered by a test.
pub fn default_tasks() -> Vec<Task> {
    let json = crate::json_strict::parse(include_str!("default_tasks.json").as_bytes())
        .unwrap_or(Json::Null);
    tasks_from_json(&json).unwrap_or_default()
}

/// Read the tasks of a manifest (`tasks[].match`), strictly.
///
/// `max_options` is the number of answer letters the head has.
///
/// # Errors
/// [`Incompat`] naming the field: unknown or missing keys, a layout that does not fit the kind,
/// too many options, two tasks with the same question.
pub fn parse(tasks: &[Task], max_options: usize) -> Result<Vec<TaskSpec>, Incompat> {
    let mut out: Vec<TaskSpec> = Vec::with_capacity(tasks.len());
    for (i, task) in tasks.iter().enumerate() {
        let mp = format!("tasks[{i}].match");
        let mut m = Obj::new(&task.matching, &mp)?;
        let qjson = m.req("question")?;
        let pjson = m.req("prompt")?;
        m.finish()?;

        let qp = format!("{mp}.question");
        let mut q = Obj::new(qjson, &qp)?;
        let rule = match task.kind.as_str() {
            "choice" => {
                let instructions = q.str("instructions")?.to_string();
                let mut options = Vec::new();
                for (j, item) in q.arr("options")?.iter().enumerate() {
                    let op = format!("{qp}.options[{j}]");
                    let mut o = Obj::new(item, &op)?;
                    let key = o.str("key")?.to_string();
                    let description = match o.opt("description") {
                        None | Some(Json::Null) => None,
                        Some(d) => Some(text(d, &format!("{op}.description"))?),
                    };
                    o.finish()?;
                    options.push((key, description));
                }
                if options.is_empty() || options.len() > max_options {
                    return Err(bad(
                        &format!("{qp}.options"),
                        &format!(
                            "has {} options; a task needs between 1 and {max_options} (the answer letters of the head)",
                            options.len()
                        ),
                    ));
                }
                Rule::Choice {
                    instructions,
                    options,
                }
            }
            "yes_no" => Rule::YesNo {
                true_description: q.str("true")?.to_string(),
                false_description: q.str("false")?.to_string(),
            },
            other => {
                return Err(bad(
                    &format!("tasks[{i}].kind"),
                    &format!("unknown kind {other:?}"),
                ));
            }
        };
        q.finish()?;

        let pp = format!("{mp}.prompt");
        let mut p = Obj::new(pjson, &pp)?;
        let system = p.str("system")?.to_string();
        let layout_name = p.str("layout")?.to_string();
        p.finish()?;
        let layout = Layout::parse(&layout_name).ok_or_else(|| {
            bad(
                &format!("{pp}.layout"),
                &format!(
                    "unknown layout {layout_name:?}; known: state, state-condition, state-pair"
                ),
            )
        })?;
        let fits = matches!(
            (&rule, layout),
            (Rule::Choice { .. }, Layout::State)
                | (
                    Rule::YesNo { .. },
                    Layout::StateCondition | Layout::StatePair
                )
        );
        if !fits {
            return Err(bad(
                &format!("{pp}.layout"),
                &format!(
                    "the layout {layout_name:?} does not fit a {} task",
                    task.kind
                ),
            ));
        }
        if let Some(j) = out.iter().position(|t| t.rule == rule) {
            return Err(bad(
                &qp,
                &format!("is the same question as tasks[{j}]: a question must match one task only"),
            ));
        }
        out.push(TaskSpec {
            id: task.id.clone(),
            rule,
            system,
            layout,
        });
    }
    Ok(out)
}

/// How many comparisons passed before the first difference (0 = a different kind of question).
enum Fit {
    Yes,
    No { passed: usize, why: String },
}

fn fit(task: &TaskSpec, q: &Question) -> Fit {
    match (&task.rule, q) {
        (
            Rule::Choice {
                instructions,
                options,
            },
            Question::Choice(c),
        ) => {
            if q.instructions().rendered() != instructions {
                // Only a whitespace difference counts as "almost this task"; anything else is a
                // different question and the answer is the list of tasks.
                let almost = q.instructions().rendered().trim() == instructions.trim();
                return Fit::No {
                    passed: usize::from(almost),
                    why: format!(
                        "its instructions differ from the ones of task {}: expected {instructions:?}",
                        task.id
                    ),
                };
            }
            if c.options().len() != options.len() {
                return Fit::No {
                    passed: 1,
                    why: format!(
                        "it has {} options but task {} has {}",
                        c.options().len(),
                        task.id,
                        options.len()
                    ),
                };
            }
            for (j, ((key, want), got)) in options.iter().zip(c.options()).enumerate() {
                if &got.key != key {
                    return Fit::No {
                        passed: 2,
                        why: format!(
                            "option {j} has the key {:?}, expected {key:?} (task {})",
                            got.key, task.id
                        ),
                    };
                }
                let given = got
                    .description
                    .as_ref()
                    .map(|d| d.rendered().to_string())
                    .filter(|d| !d.is_empty());
                if let (Some(given), Some(want)) = (&given, want)
                    && given != want
                {
                    return Fit::No {
                        passed: 3,
                        why: format!(
                            "the description of option {j} is {given:?}, expected {want:?} (task {})",
                            task.id
                        ),
                    };
                }
                if given.is_some() && want.is_none() {
                    return Fit::No {
                        passed: 3,
                        why: format!("option {j} has a description but task {} has none", task.id),
                    };
                }
            }
            Fit::Yes
        }
        (
            Rule::YesNo {
                true_description,
                false_description,
            },
            Question::YesNo(y),
        ) => {
            let got = |d: Option<&crate::schema::TextEntry>| d.map(|d| d.rendered().to_string());
            let (t, f) = (got(y.true_description()), got(y.false_description()));
            if t.as_deref() != Some(true_description.as_str()) {
                return Fit::No {
                    passed: 0,
                    why: format!(
                        "the true description is {t:?}, expected {true_description:?} (task {})",
                        task.id
                    ),
                };
            }
            if f.as_deref() != Some(false_description.as_str()) {
                return Fit::No {
                    passed: 1,
                    why: format!(
                        "the false description is {f:?}, expected {false_description:?} (task {})",
                        task.id
                    ),
                };
            }
            Fit::Yes
        }
        _ => Fit::No {
            passed: 0,
            why: String::new(),
        },
    }
}

fn describe(task: &TaskSpec) -> String {
    match &task.rule {
        Rule::Choice {
            instructions,
            options,
        } => format!(
            "{} (Choice, instructions {instructions:?}, option keys {})",
            task.id,
            options
                .iter()
                .map(|(k, _)| format!("{k:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Rule::YesNo {
            true_description,
            false_description,
        } => format!(
            "{} (YesNo, true {true_description:?}, false {false_description:?}, any instructions as the second text)",
            task.id
        ),
    }
}

/// Find the task a question corresponds to, and check the parts of the request that the prompt of
/// that task needs (the state and, for yes/no tasks, the instructions are plain text).
///
/// # Errors
/// [`Unsupported::TaskMismatch`] saying what differs, or listing the declared tasks.
pub fn find<'a>(
    tasks: &'a [TaskSpec],
    model: &str,
    name: &str,
    question: &Question,
    state: &State,
) -> Result<&'a TaskSpec, Unsupported> {
    let mismatch = |detail: String| Unsupported::TaskMismatch {
        question: name.to_string(),
        model: model.to_string(),
        detail,
    };
    let mut best: Option<(usize, String)> = None;
    let mut found: Option<&TaskSpec> = None;
    for t in tasks {
        match fit(t, question) {
            Fit::Yes => {
                found = Some(t);
                break;
            }
            Fit::No { passed, why } if passed > 0 && !why.is_empty() => {
                if best.as_ref().is_none_or(|(p, _)| passed > *p) {
                    best = Some((passed, why));
                }
            }
            Fit::No { .. } => {}
        }
    }
    let Some(task) = found else {
        return Err(mismatch(match best {
            Some((_, why)) => format!("it looks like a declared task but {why}"),
            None => format!(
                "declared tasks: {}",
                tasks.iter().map(describe).collect::<Vec<_>>().join("; ")
            ),
        }));
    };
    if !matches!(state.value(), StateValue::Str(_)) {
        return Err(mismatch(format!(
            "the state of task {} must be plain text (a string), got {}",
            task.id,
            state.value().kind()
        )));
    }
    if matches!(task.rule, Rule::YesNo { .. }) {
        let entry = question.instructions();
        let plain = matches!(entry.value(), StateValue::Str(s) if !s.trim().is_empty());
        if !plain {
            return Err(mismatch(format!(
                "the instructions of task {} are the second text of the prompt and must be plain non-empty text",
                task.id
            )));
        }
    }
    Ok(task)
}
