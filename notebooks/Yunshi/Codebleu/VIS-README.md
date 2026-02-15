Visualizing CodeBLEU vs. Objective Distance
Overview
This visualization is designed to analyze the correlation between code quality (measured via CodeBLEU) and functional performance (measured via Objective Distance). It helps determine if "standard-looking" code actually leads to better task results.

Plot Interpretation Guide
1. The Axes
X-axis (CodeBLEU Similarity): Represents the syntactic and logical similarity between the generated code and the reference code.

A value closer to 1.0 indicates the code is more "standard" or matches the reference closely.

Y-axis (Objective Distance): Represents the gap between the execution result of the generated code and the target objective.

A value closer to 0.0 indicates the code "ran correctly" or achieved high performance.

2. Understanding the Data Points (Blue Dots)
Each blue dot represents a specific experiment or a single generated code sample. Their position tells a story:

Bottom-Right Cluster: The Ideal State. These codes are both similar to the reference and functionally correct.

Top-Left Cluster: The Failure State. These codes are neither similar to the reference nor do they produce the correct output.

Bottom-Left Cluster: The "Alternative Solutions". These are interesting cases where the code does not look like the reference (low CodeBLEU) but still achieves the correct result (low Distance). This suggests the task may have multiple valid logic paths.

3. Regression Line & Confidence Interval
Red Line (Regression Line): This shows the general trend. A downward slope suggests that as CodeBLEU similarity increases, the objective distance decreases (performance improves).

Red Shaded Area (95% Confidence Interval):

Definition: This represents the level of "certainty" the model has about the trend. It shows the range where the true regression line likely lies.

Width Interpretation: * Wider Area: Indicates sparse data points. The model is less certain about the trend in that specific range (e.g., the left side of your plot).

Narrower Area: Indicates a high density of data points. The trend prediction is more reliable here (e.g., the right side of your plot).

Statistical Significance: If the shaded area is wide enough to contain a completely flat horizontal line, it implies the correlation might not be statistically significant.