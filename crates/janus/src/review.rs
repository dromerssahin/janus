//! In-memory fictional review checks and aggregate-only rating summaries.
//!
//! This module is deliberately independent of clinical interpretation. It
//! validates presentation hygiene and produces counts without serializing rows
//! or ratings.
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewRow {
    /// Internal grouping key; callers must keep this in memory.
    pub patient_key: String,
    pub source: String,
    pub fields: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewPresentation {
    pub protocol_version: String,
    pub blinded: bool,
    pub allowed_fields: BTreeSet<String>,
    pub prohibited_fields: BTreeSet<String>,
    pub reserved_patient_keys: HashSet<String>,
    pub rows: Vec<ReviewRow>,
}

impl ReviewPresentation {
    pub fn validate(&self) -> Result<(), Error> {
        if self.protocol_version.trim().is_empty() || !self.blinded || self.rows.is_empty() {
            return Err(Error::Input(
                "review presentation is not blinded or is empty".into(),
            ));
        }
        if self.allowed_fields.is_empty()
            || self.allowed_fields.iter().any(|f| f.trim().is_empty())
            || self
                .allowed_fields
                .intersection(&self.prohibited_fields)
                .next()
                .is_some()
        {
            return Err(Error::Input("invalid review field policy".into()));
        }
        let mut keys = HashSet::new();
        for row in &self.rows {
            if row.patient_key.trim().is_empty()
                || row.source.trim().is_empty()
                || !keys.insert(&row.patient_key)
                || self.reserved_patient_keys.contains(&row.patient_key)
                || row.fields.keys().any(|field| {
                    !self.allowed_fields.contains(field) || self.prohibited_fields.contains(field)
                })
                || self
                    .allowed_fields
                    .iter()
                    .any(|field| !row.fields.contains_key(field))
            {
                return Err(Error::Input(
                    "review row violates grouping, reservation, or field policy".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewRating {
    pub row_index: usize,
    pub reviewer: String,
    pub severity: String,
    pub reason: String,
    pub insufficient_information: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewAggregate {
    pub rows_presented: usize,
    pub ratings_received: usize,
    pub reviewer_count: usize,
    pub disagreement_rows: usize,
    pub suppressed_cells: usize,
    pub severity_counts: BTreeMap<String, usize>,
    pub reason_counts: BTreeMap<String, usize>,
    pub insufficient_information: usize,
}

pub fn aggregate_ratings(
    presentation: &ReviewPresentation,
    ratings: &[ReviewRating],
    minimum_cell_size: usize,
) -> Result<ReviewAggregate, Error> {
    presentation.validate()?;
    if minimum_cell_size == 0
        || ratings
            .iter()
            .any(|r| r.row_index >= presentation.rows.len())
    {
        return Err(Error::Input(
            "invalid review rating or suppression threshold".into(),
        ));
    }
    let mut reviewers = BTreeSet::new();
    let mut severity = BTreeMap::new();
    let mut reasons = BTreeMap::new();
    let mut per_row: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut insufficient_information = 0;
    for rating in ratings {
        if rating.reviewer.trim().is_empty()
            || rating.severity.trim().is_empty()
            || rating.reason.trim().is_empty()
        {
            return Err(Error::Input("review rating has an empty field".into()));
        }
        reviewers.insert(rating.reviewer.clone());
        *severity.entry(rating.severity.clone()).or_default() += 1;
        *reasons.entry(rating.reason.clone()).or_default() += 1;
        per_row
            .entry(rating.row_index)
            .or_default()
            .insert(rating.severity.clone());
        insufficient_information += usize::from(rating.insufficient_information);
    }
    let disagreement_rows = per_row.values().filter(|values| values.len() > 1).count();
    let suppressed_cells = severity
        .values()
        .filter(|count| **count < minimum_cell_size)
        .count()
        + reasons
            .values()
            .filter(|count| **count < minimum_cell_size)
            .count();
    Ok(ReviewAggregate {
        rows_presented: presentation.rows.len(),
        ratings_received: ratings.len(),
        reviewer_count: reviewers.len(),
        disagreement_rows,
        suppressed_cells,
        severity_counts: severity,
        reason_counts: reasons,
        insufficient_information,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn presentation() -> ReviewPresentation {
        let fields = BTreeMap::from([("wfns".into(), "5".into()), ("mrs".into(), "NA".into())]);
        ReviewPresentation {
            protocol_version: "fictional-v1".into(),
            blinded: true,
            allowed_fields: BTreeSet::from(["wfns".into(), "mrs".into()]),
            prohibited_fields: BTreeSet::from(["record_id".into(), "model".into()]),
            reserved_patient_keys: HashSet::from(["reserved".into()]),
            rows: vec![ReviewRow {
                patient_key: "p1".into(),
                source: "synthetic".into(),
                fields,
            }],
        }
    }

    #[test]
    fn fictional_review_aggregates_missingness_disagreement_and_small_cells() {
        let p = presentation();
        let ratings = vec![
            ReviewRating {
                row_index: 0,
                reviewer: "r1".into(),
                severity: "none".into(),
                reason: "missing_followup".into(),
                insufficient_information: true,
            },
            ReviewRating {
                row_index: 0,
                reviewer: "r2".into(),
                severity: "major".into(),
                reason: "unusual_combination".into(),
                insufficient_information: false,
            },
        ];
        let a = aggregate_ratings(&p, &ratings, 2).unwrap();
        assert_eq!(
            (a.rows_presented, a.reviewer_count, a.disagreement_rows),
            (1, 2, 1)
        );
        assert_eq!(a.suppressed_cells, 4);
        assert_eq!(a.insufficient_information, 1);
    }

    #[test]
    fn presentation_rejects_duplicates_reserved_and_prohibited_fields() {
        let mut p = presentation();
        p.rows.push(p.rows[0].clone());
        assert!(p.validate().is_err());
        p.rows.pop();
        p.rows[0].patient_key = "reserved".into();
        assert!(p.validate().is_err());
        p.rows[0].patient_key = "p1".into();
        p.rows[0].fields.insert("model".into(), "forest".into());
        assert!(p.validate().is_err());
    }
}
