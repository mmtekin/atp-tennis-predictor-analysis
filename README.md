# ATP Tennis Match Prediction Analysis

By Malik Tekin (mtekin@umich.edu)

An analysis of ATP tennis matches to predict match outcomes and explore patterns in professional tennis. This project applies practical data science techniques to understand and predict tennis match results using historical ATP tour data.

## Introduction

This project analyzes historical ATP tennis match data (after transformation, 6152 rows since every row from original dataset had to be split into 2 rows) to uncover patterns in professional tennis and build a pre‑match predictor of match outcomes. We explore how variables like player ranking, age, and Elo rating affect winning probability, then train a model that—using only these pre‑match stats—forecasts which player will win. Such insights can aid coaches, bettors, and tennis analysts in understanding match dynamics before a ball is struck.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning Process

The data cleaning process involved several sophisticated transformations to prepare the ATP match data for analysis:

1. **Elo Rating Computation**
   - Implemented a dynamic Elo rating system for all players
   - Processed matches chronologically to maintain accurate rating progression
   - Initial rating of 1500 for new players
   - Ratings update after each match using standard Elo formula with k=32

2. **Data Restructuring**
   - Transformed the original match-centric format to a player-centric format
   - Each match was split into two rows to represent both perspectives (winner and loser)
   - This doubled our dataset size but provides balanced training data for our prediction model

3. **Feature Engineering**
   - Converted tournament dates to datetime format
   - Standardized player attributes (rank, age, hand, rank points)
   - Created symmetric features for both players (p1 and p2)
   - Added binary outcome variable 'did_p1_win'

4. **Final Dataset Structure**
Key features in transformed dataset:
   - Tournament info: date, surface, draw_size, level, round
   - Player 1 stats: rank, age, hand, rank_points, elo
   - Player 2 stats: rank, age, hand, rank_points, elo
   - Outcome: did_p1_win (binary)

This transformation approach ensures our model can learn from both winning and losing perspectives, while maintaining the temporal nature of the Elo rating system.

### Univariate Analysis

Our univariate analysis revealed several interesting distributions in our tennis match dataset:

1. **Player Rankings Distribution**
<iframe
src="assets/rankingsDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The distribution of player rankings shows a significant skew towards higher-ranked players (ranks 1-49), with 2,778 matches compared to 1,831 matches for players ranked 50-99. This makes sense as our dataset focuses on major tournaments rather than challenger events.

2. **Age Distribution**
<iframe
src="assets/ageDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The age distribution reveals that most players are between 25-29 years old, with a steep drop-off after age 29. This suggests that players' physical prime typically lasts until around age 29.

3. **Court Surface Distribution**
<iframe
src="assets/courtDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

Hard courts dominate the dataset, followed by clay, with grass courts having the fewest matches. This distribution will be important when evaluating our model's performance across different surfaces.

4. **Player Handedness**
<iframe
src="assets/handDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The vast majority of players are right-handed, which aligns with general population statistics. This distribution is particularly interesting given the traditional belief about left-handed players having an advantage.

### Bivariate Analysis

Our bivariate analysis focused on understanding how different factors relate to match outcomes:

1. **Age Difference and Win Rate**
<iframe
src="assets/winRateAgeDiffBin.html"
width="800"
height="600"
frameborder="0"
></iframe>

This analysis reveals that:
- Players tend to have higher win rates when they are 2-8 years younger than their opponents
- A sharp increase in win rate occurs with >10 years age difference
- Being older (especially by >4 years) correlates with lower win rates
- The most significant drop in win rate occurs when a player is >12 years older

2. **Handedness and Win Rate**
<iframe
src="assets/winRatePlayerHandBar.html"
width="800"
height="600"
frameborder="0"
></iframe>

Surprisingly, our analysis shows no significant advantage for left-handed players, contrary to common tennis wisdom. This suggests that handedness might not be a crucial feature for match prediction.

3. **Ranking Difference and Win Rate**
<iframe
src="assets/winRateByRankDiff.html"
width="800"
height="600"
frameborder="0"
></iframe>

The relationship between ranking difference and win probability is strong and clear:
- Players with better rankings (negative rank difference) win approximately 75% of matches
- Equal rankings (~0 difference) correspond to ~50% win rates
- Lower-ranked players (positive rank difference) win less than 25% of matches against opponents ranked 90+ spots higher

This strong correlation makes ranking difference a potentially powerful predictor for our model.

### Interesting Aggregates

We analyzed how player ranking tiers interact with court surfaces to influence win rates:

<iframe
src="assets/winRateBySurfaceAndRankTier.html"
width="800"
height="400"
frameborder="0"
></iframe>

This pivot table reveals several fascinating patterns about the relationship between player ranking and surface-specific performance:

1. **Top 10 Players' Dominance**:
   - Elite players (Top 10) maintain impressive win rates >70% across all surfaces
   - Highest win rates on grass (77%) and clay (76%)
   - Relatively lower win rate on hard courts (71%), suggesting surface might be a slight equalizer

2. **Mid-Tier Performance (11-50)**:
   - Players ranked 11-25 show consistent performance (~60% win rate) across surfaces
   - Rankings 26-50 maintain win rates around 48-51%, with grass courts showing slightly better outcomes

3. **Lower Rankings Impact**:
   - Players ranked 100+ struggle significantly with ~40% win rates across all surfaces
   - The skill gap is most pronounced on clay courts, where ranking appears to be particularly important

This analysis suggests that while top players maintain their advantage across all surfaces, the impact of ranking on win probability varies by surface type. Hard courts, being the most common surface, show slightly more competitive matches across ranking tiers, possibly due to players' greater familiarity with the surface.

### Handling Missing Values

Our data cleaning process was designed to minimize the impact of missing values:

1. **Elo Ratings**
   - New players automatically assigned 1500
   - Ratings evolve naturally through match play
   - No imputation needed as system is self-contained

2. **Player Attributes**
   - Rank and rank points: Used as-is (missing values indicate unranked players)
   - Age and hand: Complete in our dataset
   - Surface: No missing values in selected tournaments

3. **Tournament Information**
   - All selected fields (date, surface, level, round) are complete
   - Draw size is consistently recorded

The transformation process ensured that our final dataset contains no missing values that would impact our prediction task.

## Framing a Prediction Problem

[Your prediction problem description will go here]

## Baseline Model

[Your baseline model description and results will go here]

## Final Model

[Your final model description, improvements, and results will go here]

---
*This project was created as part of EECS 398: Practical Data Science at the University of Michigan.*
