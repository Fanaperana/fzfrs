//! # fzfrs - Fuzzy Search Algorithm Implementation
//!
//! A comprehensive, educational implementation of fuzzy search algorithms.
//! This library provides tools for approximate string matching using the
//! Levenshtein distance algorithm.
//!
//! ## Quick Start
//!
//! ```rust
//! use fzfrs::{FuzzySearcher, LevenshteinDistance};
//!
//! // Calculate edit distance between two strings
//! let distance = LevenshteinDistance::compute("kitten", "sitting");
//! assert_eq!(distance, 3);
//!
//! // Get similarity score (0.0 to 1.0)
//! let similarity = LevenshteinDistance::similarity("apple", "aple");
//! assert!(similarity >= 0.8);
//!
//! // Search through candidates
//! let searcher = FuzzySearcher::new(0.6);
//! let candidates = vec!["apple", "application", "banana"];
//! let results = searcher.search("aple", &candidates);
//! ```
//!
//! ## Algorithm Overview
//!
//! The Levenshtein distance between two strings is the minimum number of
//! single-character edits (insertions, deletions, or substitutions) required
//! to change one string into the other.
//!
//! This implementation uses the Wagner-Fischer dynamic programming algorithm
//! with an optional memory-optimized variant that uses only O(min(m,n)) space.

use std::cmp::min;
use std::fmt;

// ============================================================================
// LEVENSHTEIN DISTANCE - Core Algorithm Implementation
// ============================================================================

/// Core implementation of the Levenshtein distance algorithm.
///
/// The Levenshtein distance (also known as edit distance) measures the minimum
/// number of single-character edits needed to transform one string into another.
///
/// ## Operations
///
/// | Operation    | Description           | Example                    |
/// |--------------|-----------------------|----------------------------|
/// | Insert       | Add a character       | `cat` → `cart` (insert r)  |
/// | Delete       | Remove a character    | `cart` → `cat` (delete r)  |
/// | Substitute   | Replace a character   | `cat` → `bat` (sub c → b)  |
///
/// ## Example
///
/// ```rust
/// use fzfrs::LevenshteinDistance;
///
/// let distance = LevenshteinDistance::compute("kitten", "sitting");
/// assert_eq!(distance, 3); // k→s, e→i, insert g
///
/// let similarity = LevenshteinDistance::similarity("apple", "apple");
/// assert_eq!(similarity, 1.0); // Identical strings
/// ```
pub struct LevenshteinDistance;

impl LevenshteinDistance {
    /// Computes the Levenshtein distance between two strings.
    ///
    /// Uses the standard Wagner-Fischer algorithm with O(m × n) time and space.
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION levenshtein_distance(source, target):
    ///     m = length(source)
    ///     n = length(target)
    ///     
    ///     IF m == 0: RETURN n
    ///     IF n == 0: RETURN m
    ///     
    ///     matrix = new Matrix[m+1][n+1]
    ///     
    ///     // Initialize first column and row
    ///     FOR i FROM 0 TO m: matrix[i][0] = i
    ///     FOR j FROM 0 TO n: matrix[0][j] = j
    ///     
    ///     // Fill the matrix
    ///     FOR i FROM 1 TO m:
    ///         FOR j FROM 1 TO n:
    ///             cost = IF source[i-1] == target[j-1] THEN 0 ELSE 1
    ///             matrix[i][j] = MIN(
    ///                 matrix[i-1][j] + 1,      // deletion
    ///                 matrix[i][j-1] + 1,      // insertion
    ///                 matrix[i-1][j-1] + cost  // substitution
    ///             )
    ///     
    ///     RETURN matrix[m][n]
    /// ```
    ///
    /// ## Parameters
    ///
    /// * `source` - The source string
    /// * `target` - The target string to compare against
    ///
    /// ## Returns
    ///
    /// The minimum number of edits required to transform source into target.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// assert_eq!(LevenshteinDistance::compute("", "abc"), 3);
    /// assert_eq!(LevenshteinDistance::compute("abc", ""), 3);
    /// assert_eq!(LevenshteinDistance::compute("abc", "abc"), 0);
    /// assert_eq!(LevenshteinDistance::compute("kitten", "sitting"), 3);
    /// ```
    pub fn compute(source: &str, target: &str) -> usize {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len();
        let n = target_chars.len();

        // Handle edge cases: if one string is empty,
        // distance equals the length of the other
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Create the distance matrix of size (m+1) x (n+1)
        // matrix[i][j] represents the edit distance between
        // source[0..i] and target[0..j]
        let mut matrix = vec![vec![0usize; n + 1]; m + 1];

        // STEP 1: Initialize first column
        // Transforming source[0..i] to empty string requires i deletions
        for i in 0..=m {
            matrix[i][0] = i;
        }

        // STEP 2: Initialize first row
        // Transforming empty string to target[0..j] requires j insertions
        for j in 0..=n {
            matrix[0][j] = j;
        }

        // STEP 3: Fill in the matrix using dynamic programming
        for i in 1..=m {
            for j in 1..=n {
                // Cost is 0 if characters match, 1 if substitution needed
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0
                } else {
                    1
                };

                // Take the minimum of three possible operations:
                let deletion = matrix[i - 1][j] + 1; // Delete from source
                let insertion = matrix[i][j - 1] + 1; // Insert into source
                let substitution = matrix[i - 1][j - 1] + cost; // Substitute (or match)

                matrix[i][j] = min(deletion, min(insertion, substitution));
            }
        }

        // STEP 4: Result is in the bottom-right cell
        matrix[m][n]
    }

    /// Memory-optimized version using only two rows.
    ///
    /// This version uses O(min(m,n)) space instead of O(m × n).
    /// Useful for comparing very long strings.
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION levenshtein_optimized(source, target):
    ///     m = length(source)
    ///     n = length(target)
    ///     
    ///     IF m == 0: RETURN n
    ///     IF n == 0: RETURN m
    ///     
    ///     // Ensure we use the shorter string for columns (less memory)
    ///     IF m < n: SWAP(source, target); SWAP(m, n)
    ///     
    ///     previous_row = [0, 1, 2, ..., n]
    ///     current_row = new Array[n+1]
    ///     
    ///     FOR i FROM 1 TO m:
    ///         current_row[0] = i
    ///         FOR j FROM 1 TO n:
    ///             cost = IF source[i-1] == target[j-1] THEN 0 ELSE 1
    ///             current_row[j] = MIN(
    ///                 current_row[j-1] + 1,    // insertion
    ///                 previous_row[j] + 1,      // deletion
    ///                 previous_row[j-1] + cost  // substitution
    ///             )
    ///         SWAP(previous_row, current_row)
    ///     
    ///     RETURN previous_row[n]
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// // Same result as compute(), but uses less memory
    /// let distance = LevenshteinDistance::compute_optimized("kitten", "sitting");
    /// assert_eq!(distance, 3);
    /// ```
    pub fn compute_optimized(source: &str, target: &str) -> usize {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len();
        let n = target_chars.len();

        // Handle edge cases
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Optimization: Use shorter string for columns to minimize memory
        let (source_chars, target_chars, m, n) = if m < n {
            (target_chars, source_chars, n, m)
        } else {
            (source_chars, target_chars, m, n)
        };

        // Only need two rows instead of full matrix
        let mut previous_row: Vec<usize> = (0..=n).collect();
        let mut current_row: Vec<usize> = vec![0; n + 1];

        // Process each character of source
        for i in 1..=m {
            // First cell of current row = number of deletions needed
            current_row[0] = i;

            for j in 1..=n {
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0
                } else {
                    1
                };

                current_row[j] = min(
                    current_row[j - 1] + 1,     // Insertion
                    min(
                        previous_row[j] + 1,     // Deletion
                        previous_row[j - 1] + cost, // Substitution
                    ),
                );
            }

            // Swap rows for next iteration
            std::mem::swap(&mut previous_row, &mut current_row);
        }

        // Result is in previous_row after the swap
        previous_row[n]
    }

    /// Computes the similarity score between two strings.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0 (identical).
    ///
    /// ## Formula
    ///
    /// ```text
    /// similarity = 1.0 - (distance / max_length)
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// assert_eq!(LevenshteinDistance::similarity("abc", "abc"), 1.0);
    /// assert_eq!(LevenshteinDistance::similarity("", ""), 1.0);
    ///
    /// let sim = LevenshteinDistance::similarity("kitten", "sitting");
    /// assert!((sim - 0.571).abs() < 0.01); // ~57.1%
    /// ```
    pub fn similarity(source: &str, target: &str) -> f64 {
        let source_len = source.chars().count();
        let target_len = target.chars().count();
        let max_len = source_len.max(target_len);

        // Two empty strings are considered identical
        if max_len == 0 {
            return 1.0;
        }

        let distance = Self::compute_optimized(source, target);

        // Convert distance to similarity score
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Case-insensitive similarity calculation.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// let sim = LevenshteinDistance::similarity_ignore_case("Apple", "APPLE");
    /// assert_eq!(sim, 1.0);
    /// ```
    pub fn similarity_ignore_case(source: &str, target: &str) -> f64 {
        Self::similarity(&source.to_lowercase(), &target.to_lowercase())
    }
}

// ============================================================================
// EDIT OPERATIONS - Tracking what changes are needed
// ============================================================================

/// Represents a single edit operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOperation {
    /// Keep the character (no change needed)
    Keep(char),
    /// Insert a character
    Insert(char),
    /// Delete a character
    Delete(char),
    /// Substitute one character for another
    Substitute { from: char, to: char },
}

impl fmt::Display for EditOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditOperation::Keep(c) => write!(f, "keep '{}'", c),
            EditOperation::Insert(c) => write!(f, "insert '{}'", c),
            EditOperation::Delete(c) => write!(f, "delete '{}'", c),
            EditOperation::Substitute { from, to } => write!(f, "replace '{}' → '{}'", from, to),
        }
    }
}

/// Detailed result of comparing two strings.
#[derive(Debug, Clone)]
pub struct EditResult {
    /// The source string
    pub source: String,
    /// The target string
    pub target: String,
    /// The edit distance
    pub distance: usize,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// The sequence of operations to transform source into target
    pub operations: Vec<EditOperation>,
}

impl EditResult {
    /// Returns a human-readable description of the match quality.
    pub fn quality(&self) -> &'static str {
        match (self.similarity * 100.0) as u32 {
            95..=100 => "Excellent",
            85..=94 => "Very Good",
            70..=84 => "Good",
            50..=69 => "Fair",
            _ => "Poor",
        }
    }
}

impl fmt::Display for EditResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Comparison: '{}' → '{}'", self.source, self.target)?;
        writeln!(f, "Distance: {} operations", self.distance)?;
        writeln!(f, "Similarity: {:.1}% ({})", self.similarity * 100.0, self.quality())?;
        if !self.operations.is_empty() {
            writeln!(f, "Operations:")?;
            for (i, op) in self.operations.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, op)?;
            }
        }
        Ok(())
    }
}

/// Extended Levenshtein operations that also track the edit path.
pub struct LevenshteinWithOperations;

impl LevenshteinWithOperations {
    /// Computes the edit distance and returns detailed information including
    /// the sequence of operations needed.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinWithOperations;
    ///
    /// let result = LevenshteinWithOperations::compute("cat", "cart");
    /// println!("{}", result);
    /// // Shows: insert 'r'
    /// ```
    pub fn compute(source: &str, target: &str) -> EditResult {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len();
        let n = target_chars.len();

        // Edge cases
        if m == 0 && n == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: 0,
                similarity: 1.0,
                operations: vec![],
            };
        }

        if m == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: n,
                similarity: 0.0,
                operations: target_chars.iter().map(|&c| EditOperation::Insert(c)).collect(),
            };
        }

        if n == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: m,
                similarity: 0.0,
                operations: source_chars.iter().map(|&c| EditOperation::Delete(c)).collect(),
            };
        }

        // Build the distance matrix
        let mut matrix = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m {
            matrix[i][0] = i;
        }
        for j in 0..=n {
            matrix[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0
                } else {
                    1
                };

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,
                    min(matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost),
                );
            }
        }

        // Backtrack to find the operations
        let operations = Self::backtrack(&matrix, &source_chars, &target_chars);

        let distance = matrix[m][n];
        let max_len = m.max(n);
        let similarity = 1.0 - (distance as f64 / max_len as f64);

        EditResult {
            source: source.to_string(),
            target: target.to_string(),
            distance,
            similarity,
            operations,
        }
    }

    /// Backtracks through the matrix to find the sequence of operations.
    fn backtrack(
        matrix: &[Vec<usize>],
        source_chars: &[char],
        target_chars: &[char],
    ) -> Vec<EditOperation> {
        let mut operations = Vec::new();
        let mut i = source_chars.len();
        let mut j = target_chars.len();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && source_chars[i - 1] == target_chars[j - 1] {
                // Characters match - no operation needed (keep)
                operations.push(EditOperation::Keep(source_chars[i - 1]));
                i -= 1;
                j -= 1;
            } else if i > 0 && j > 0 && matrix[i][j] == matrix[i - 1][j - 1] + 1 {
                // Substitution
                operations.push(EditOperation::Substitute {
                    from: source_chars[i - 1],
                    to: target_chars[j - 1],
                });
                i -= 1;
                j -= 1;
            } else if j > 0 && matrix[i][j] == matrix[i][j - 1] + 1 {
                // Insertion
                operations.push(EditOperation::Insert(target_chars[j - 1]));
                j -= 1;
            } else if i > 0 && matrix[i][j] == matrix[i - 1][j] + 1 {
                // Deletion
                operations.push(EditOperation::Delete(source_chars[i - 1]));
                i -= 1;
            } else {
                // This shouldn't happen with a valid matrix
                break;
            }
        }

        // Reverse since we backtracked from the end
        operations.reverse();

        // Filter out Keep operations for cleaner output
        operations
            .into_iter()
            .filter(|op| !matches!(op, EditOperation::Keep(_)))
            .collect()
    }
}

// ============================================================================
// FUZZY SEARCHER - High-level search interface
// ============================================================================

/// A match result from fuzzy searching.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// The matched text
    pub text: String,
    /// Similarity score (0.0 to 1.0)
    pub score: f64,
    /// The original index in the candidates list
    pub index: usize,
}

impl MatchResult {
    /// Returns a human-readable quality description.
    pub fn quality(&self) -> &'static str {
        match (self.score * 100.0) as u32 {
            95..=100 => "Excellent",
            85..=94 => "Very Good",
            70..=84 => "Good",
            50..=69 => "Fair",
            _ => "Poor",
        }
    }
}

/// Configuration for the fuzzy searcher.
#[derive(Debug, Clone)]
pub struct FuzzySearcher {
    /// Minimum similarity threshold (0.0 to 1.0)
    threshold: f64,
    /// Whether to ignore case when comparing
    case_insensitive: bool,
    /// Maximum number of results to return (None = unlimited)
    max_results: Option<usize>,
}

impl Default for FuzzySearcher {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            case_insensitive: true,
            max_results: None,
        }
    }
}

impl FuzzySearcher {
    /// Creates a new FuzzySearcher with the given similarity threshold.
    ///
    /// ## Parameters
    ///
    /// * `threshold` - Minimum similarity score (0.0 to 1.0) for a match
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.7); // 70% similarity required
    /// ```
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Enables or disables case-insensitive matching.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6)
    ///     .case_insensitive(false); // Case-sensitive matching
    /// ```
    pub fn case_insensitive(mut self, value: bool) -> Self {
        self.case_insensitive = value;
        self
    }

    /// Sets the maximum number of results to return.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6)
    ///     .max_results(10); // Return at most 10 matches
    /// ```
    pub fn max_results(mut self, limit: usize) -> Self {
        self.max_results = Some(limit);
        self
    }

    /// Searches for matches in the given candidates.
    ///
    /// Returns a list of matches sorted by score (best first).
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION fuzzy_search(query, candidates, threshold):
    ///     results = []
    ///     
    ///     FOR EACH (index, candidate) IN candidates:
    ///         score = similarity(query, candidate)
    ///         
    ///         IF score >= threshold:
    ///             ADD (candidate, score, index) TO results
    ///     
    ///     SORT results BY score DESCENDING
    ///     RETURN results
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6);
    /// let candidates = vec!["apple", "application", "banana"];
    /// let results = searcher.search("aple", &candidates);
    ///
    /// for result in results {
    ///     println!("{}: {:.1}%", result.text, result.score * 100.0);
    /// }
    /// ```
    pub fn search<S: AsRef<str>>(&self, query: &str, candidates: &[S]) -> Vec<MatchResult> {
        let query_normalized = if self.case_insensitive {
            query.to_lowercase()
        } else {
            query.to_string()
        };

        let mut results: Vec<MatchResult> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let candidate_str = candidate.as_ref();
                let candidate_normalized = if self.case_insensitive {
                    candidate_str.to_lowercase()
                } else {
                    candidate_str.to_string()
                };

                let score = LevenshteinDistance::similarity(&query_normalized, &candidate_normalized);

                if score >= self.threshold {
                    Some(MatchResult {
                        text: candidate_str.to_string(),
                        score,
                        index,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending (best matches first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Apply max_results limit if set
        if let Some(limit) = self.max_results {
            results.truncate(limit);
        }

        results
    }

    /// Finds the single best match for a query.
    ///
    /// Returns None if no match meets the threshold.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6);
    /// let candidates = vec!["apple", "banana", "orange"];
    ///
    /// if let Some(best) = searcher.find_best("aple", &candidates) {
    ///     println!("Best match: {} ({:.1}%)", best.text, best.score * 100.0);
    /// }
    /// ```
    pub fn find_best<S: AsRef<str>>(&self, query: &str, candidates: &[S]) -> Option<MatchResult> {
        self.search(query, candidates).into_iter().next()
    }

    /// Checks if a query matches a single candidate.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.7);
    ///
    /// assert!(searcher.matches("apple", "aple"));
    /// assert!(!searcher.matches("apple", "banana"));
    /// ```
    pub fn matches(&self, candidate: &str, query: &str) -> bool {
        let score = if self.case_insensitive {
            LevenshteinDistance::similarity_ignore_case(query, candidate)
        } else {
            LevenshteinDistance::similarity(query, candidate)
        };

        score >= self.threshold
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(LevenshteinDistance::compute("", ""), 0);
        assert_eq!(LevenshteinDistance::compute("abc", ""), 3);
        assert_eq!(LevenshteinDistance::compute("", "abc"), 3);
        assert_eq!(LevenshteinDistance::compute("abc", "abc"), 0);
    }

    #[test]
    fn test_levenshtein_single_operations() {
        // Single insertion
        assert_eq!(LevenshteinDistance::compute("cat", "cart"), 1);
        // Single deletion
        assert_eq!(LevenshteinDistance::compute("cart", "cat"), 1);
        // Single substitution
        assert_eq!(LevenshteinDistance::compute("cat", "bat"), 1);
    }

    #[test]
    fn test_levenshtein_complex() {
        assert_eq!(LevenshteinDistance::compute("kitten", "sitting"), 3);
        assert_eq!(LevenshteinDistance::compute("saturday", "sunday"), 3);
        assert_eq!(LevenshteinDistance::compute("algorithm", "altruistic"), 6);
    }

    #[test]
    fn test_optimized_matches_standard() {
        let test_pairs = [
            ("", ""),
            ("abc", ""),
            ("", "abc"),
            ("kitten", "sitting"),
            ("algorithm", "altruistic"),
            ("hello world", "hello word"),
        ];

        for (s1, s2) in test_pairs {
            assert_eq!(
                LevenshteinDistance::compute(s1, s2),
                LevenshteinDistance::compute_optimized(s1, s2),
                "Mismatch for '{}' vs '{}'",
                s1,
                s2
            );
        }
    }

    #[test]
    fn test_similarity() {
        assert_eq!(LevenshteinDistance::similarity("abc", "abc"), 1.0);
        assert_eq!(LevenshteinDistance::similarity("", ""), 1.0);
        assert_eq!(LevenshteinDistance::similarity("abc", "xyz"), 0.0);
        
        let sim = LevenshteinDistance::similarity("kitten", "sitting");
        assert!((sim - 0.571).abs() < 0.01);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            LevenshteinDistance::similarity_ignore_case("Apple", "APPLE"),
            1.0
        );
        assert_eq!(
            LevenshteinDistance::similarity_ignore_case("Hello", "hello"),
            1.0
        );
    }

    #[test]
    fn test_fuzzy_searcher() {
        let searcher = FuzzySearcher::new(0.6);
        let candidates = vec!["apple", "application", "applet", "banana"];

        let results = searcher.search("aple", &candidates);

        assert!(!results.is_empty());
        assert_eq!(results[0].text, "apple"); // Best match
    }

    #[test]
    fn test_fuzzy_searcher_case_insensitive() {
        let searcher = FuzzySearcher::new(0.8).case_insensitive(true);
        let candidates = vec!["Apple", "APPLE", "apple"];

        let results = searcher.search("apple", &candidates);

        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.score, 1.0);
        }
    }

    #[test]
    fn test_fuzzy_searcher_max_results() {
        let searcher = FuzzySearcher::new(0.5).max_results(2);
        let candidates = vec!["apple", "application", "applet", "app"];

        let results = searcher.search("app", &candidates);

        assert!(results.len() <= 2);
    }

    #[test]
    fn test_edit_operations() {
        let result = LevenshteinWithOperations::compute("cat", "cart");
        assert_eq!(result.distance, 1);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(result.operations[0], EditOperation::Insert('r')));
    }

    #[test]
    fn test_unicode() {
        assert_eq!(LevenshteinDistance::compute("café", "cafe"), 1);
        assert_eq!(LevenshteinDistance::compute("日本", "日本語"), 1);
        assert_eq!(LevenshteinDistance::compute("🎉", "🎊"), 1);
    }
}
