# Lecture 5: Attention

Attention is a mechanism that lets a model decide which parts of the input matter most for the current prediction.

In simple terms, attention computes a relevance score between a query and a set of keys, then uses those scores to produce a weighted combination of values.

Why it helps:

- it avoids forcing all information through a single fixed-length representation
- it lets the model focus on the most relevant tokens
- it improves performance on long-range dependencies

When comparing approaches, lecture notes often contrast recurrent models with attention-based models by showing that attention gives more direct access to relevant context.
