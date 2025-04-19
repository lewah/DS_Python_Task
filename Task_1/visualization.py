from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from mapping import DataHandling, TestDataMapper

"""
Bokeh Visualization for Training Data, Ideal Functions, and Mapped Test Data.
This module provides functions to:
1. Load data from an SQLite database.
2. Visualize best-fit ideal functions for training data.
3. Visualize training data, ideal functions, and mapped test data.
Functions:
    - load_data(db_path, train_table, ideal_table, test_table)
        Loads training, ideal, and mapped test data.
    - create_best_fit_visualization(training_df, ideal_df, best_fits)
        Generates a Bokeh scatter and line plot for best-fit ideal functions.
    - create_mapped_data_visualization(training_df, ideal_df, mapped_data)
        Generates a Bokeh scatter and line plot showing training data, ideal functions, 
        and mapped test data.
Dependencies:
    - bokeh.plotting (figure, show, output_file)
    - bokeh.io (output_notebook)
    - pandas
    - numpy
    - mapping (DataHandling, TestDataMapper)
"""

def load_data(db_path, train_table, ideal_table, test_table):
    """
    Load and return training, ideal, and mapped test data.
    Args:
        db_path (str): Path to the SQLite database.
        train_table (str): Name of the training data table.
        ideal_table (str): Name of the ideal functions data table.
        test_table (str): Name of the test data table.
    Returns:
        tuple: (training_df, ideal_df, mapped_data, best_fits)
            - training_df (pd.DataFrame): Training dataset.
            - ideal_df (pd.DataFrame): Ideal functions dataset.
            - mapped_data (pd.DataFrame): Test data mapped to ideal functions.
            - best_fits (dict): Best-fit functions {train_col: (ideal_col, MSE)}
    """
    data_handler = DataHandling(db_path, train_table, ideal_table, test_table)
    test_mapper = TestDataMapper(db_path, train_table, ideal_table, test_table)
    mapped_data = test_mapper.map_test_data() # Get mapped test data
    best_fits = test_mapper.find_best_fit() # Get the best-fit functions
    
    training_df = data_handler.train_data
    ideal_df = data_handler.ideal_data
        
    return training_df, ideal_df, mapped_data, best_fits

def create_best_fit_visualization(training_df, ideal_df, best_fits):
    """
    Generate a Bokeh plot showing training data and their best-fit ideal functions.
    Args:
        training_df (pd.DataFrame): Training dataset.
        ideal_df (pd.DataFrame): Ideal functions dataset.
        best_fits (dict): Mapping of training columns to their best-fit ideal functions.
    Visualization:
        - Training data is shown as scatter points.
        - Corresponding best-fit ideal functions are plotted as lines.
        - A legend is included with MSE values.
    Output:
        - Saves the plot as "best_fit_visualization.html".
        - Opens the plot in a browser.
    """
    output_file("best_fit_visualization.html")
    
    p = figure(
        title="Best-Fit Ideal Functions for Training Data",
        x_axis_label="x", 
        y_axis_label="y",
        width=900, 
        height=700
    )

    # Colors for each training-ideal pair
    colors = ["navy", "darkorange", "darkgreen", "crimson"]
    
    # Plot each best-fit pair
    for i, (train_col, (ideal_col, mse)) in enumerate(best_fits.items()):
        # Training data (scatter)
        p.scatter(training_df["x"], training_df[train_col], 
                 color=colors[i], legend_label=f"Training {train_col} → Ideal {ideal_col} (MSE: {mse})", 
                 size=5, alpha=0.8)
        
        # Ideal function (line)
        p.line(ideal_df["x"], ideal_df[ideal_col], 
              color=colors[i], line_width=3, alpha=0.8)

    # Customizations
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "10pt"
    p.legend.click_policy = "hide"
    show(p)

def create_mapped_data_visualization(training_df, ideal_df, mapped_data):
    """
    Generate a Bokeh visualization of training data, ideal functions, and mapped test data.
    Args:
        training_df (pd.DataFrame): Training dataset.
        ideal_df (pd.DataFrame): Ideal functions dataset.
        mapped_data (pd.DataFrame): Mapped test data.
    Visualization:
        - Training data is plotted as scatter points in different colors.
        - Best-fit ideal functions are plotted as dashed lines.
        - Mapped test data is displayed as purple scatter points.
    Output:
        - Saves the plot as "visualization.html".
        - Opens the plot in a browser.
    """
    # Use output_file for script mode (saves to HTML and opens in browser)
    output_file("visualization.html")  
    
    # Create figure
    p = figure(
        title="Training Data, Ideal Functions & Mapped Test Data",
        x_axis_label="x", 
        y_axis_label="y",
        width=800, 
        height=600
    )

    # Plot training data
    p.scatter(training_df["x"], training_df["y1"], color="blue", legend_label="Training y1", size=4, alpha=0.6)
    p.scatter(training_df["x"], training_df["y2"], color="orange", legend_label="Training y2", size=4, alpha=0.6)
    p.scatter(training_df["x"], training_df["y3"], color="green", legend_label="Training y3", size=4, alpha=0.6)
    p.scatter(training_df["x"], training_df["y4"], color="red", legend_label="Training y4", size=4, alpha=0.6)

    # Plot ideal functions
    p.line(ideal_df["x"], ideal_df["y42"], color="blue", legend_label="Ideal y42", line_width=2, line_dash="dashed")
    p.line(ideal_df["x"], ideal_df["y41"], color="orange", legend_label="Ideal y41", line_width=2, line_dash="dashed")
    p.line(ideal_df["x"], ideal_df["y11"], color="green", legend_label="Ideal y11", line_width=2, line_dash="dashed")
    p.line(ideal_df["x"], ideal_df["y48"], color="red", legend_label="Ideal y48", line_width=2, line_dash="dashed")

    # Plot mapped test data
    p.scatter(mapped_data["x"], mapped_data["y_test"], color="purple", legend_label="Mapped Test Data", size=7, marker="circle", alpha=1.0)

    # Customize legend
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "10pt"
    p.legend.click_policy = "hide"

    # Show plot (opens in browser)
    show(p)

