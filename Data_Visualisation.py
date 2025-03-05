import pandas as pd 
import plotly.graph_objects as go
import re
import numpy as np
import datetime
from dateutil.parser import parse
import umap
from plotly.io import to_json

class BasePlot:
    def __init__(self, data, colors=None, font='Arial, Helvetica, sans-serif', **kwargs):
        self.data = data
        self.colors = {
            'forest_green': '#125D0D',  # Forest Green (unchanged)
            'burnt_orange': '#A34700',  # Darkened Burnt Orange
            'peach': '#E6B8A2',         # Muted Peach (more earthy)
            'mustard_yellow': '#B58B0B',# Darkened Mustard Yellow
            'dark_grey': '#3C3C3C',     # Slightly Darker Grey
            'navy_blue': '#000060',     # Darkened Navy Blue
            'chocolate_brown': '#4A2E14', # Darkened Chocolate Brown
            'berry_red': '#732043',     # Darkened Berry Red
            'light_gray': '#A9A9A9'     # Light Gray (unchanged)
        }

        self.font = font

        # Initialize font sizes
        self.title_font_size = kwargs.get('title_font_size', 24)
        self.axis_label_font_size = kwargs.get('axis_label_font_size', 16)
        self.axis_title_font_size = kwargs.get('axis_title_font_size', 20)
        self.tick_font_size = kwargs.get('tick_font_size', 14)

        # Initialize DataPreparation
        self.data_preparation = self.DataPreparation(self.data)

        # Extract additional columns if present
        self.topic_column = self.data_preparation.identify_column('BERTopic_Topic')
        self.topic_words_column = self.data_preparation.identify_column('BERTopic_Top_Words')

        # Extract metadata information
        self.metadata_dict = self.data_preparation.extract_metadata()
        print(f"Extracted metadata: {self.metadata_dict}")

        # Extract metadata fields
        self.your_dataset = self.metadata_dict.get('your_dataset', '')
        self.our_dataset = self.metadata_dict.get('our_datasets', '')
        self.background_dataset_size = self.metadata_dict.get('background dataset size', '')
        self.sentiment_context = self.metadata_dict.get('sentiment context', '')
        self.societally_linear = self.metadata_dict.get('societally linear', '')
        self.perplexity = self.metadata_dict.get('perplexity', '30')
        self.learning_rate = self.metadata_dict.get('learning rate', '200')
        print(f"Your dataset: {self.your_dataset}, Our dataset: {self.our_dataset}")
        print(f"Sentiment context: {self.sentiment_context}")

    class DataPreparation:
        def __init__(self, data):
            self.data = data
            self.text_column = self.identify_column('Text')
            self.source_column = self.identify_column('Source')
            self.time_column = self.identify_column('Time')
            self.prediction_columns = self.identify_prediction_columns()
            self.metadata_column = self.identify_column('Metadata')
            self.time_handling = self.TimeHandling(self)
            self.text_preparation = self.TextPreparation()

        def identify_column(self, keyword):
            # Find the column that matches the keyword, ignoring case and extra spaces
            for col in self.data.columns:
                if keyword.lower() == col.strip().lower():
                    return col
            print(f"No column found for keyword '{keyword}'")
            return None

        def identify_prediction_columns(self):
            prediction_cols = []
            for col in self.data.columns:
                if 'continuous:' in col.lower(): 
                    prediction_cols.append(col)
                elif 'probability distribution' in col.lower():
                    prediction_cols.append(col)
            return prediction_cols

        def extract_prediction_info(self, column_name):
            if 'continuous:' in column_name.lower():
                prediction_word = column_name.split('continuous:')[1].strip()
                prediction_type = 'continuous'
                number = None
            elif 'probability distribution' in column_name.lower():
                # Use regular expression to extract number and word
                match = re.match(r'probability distribution\s*(\d+):\s*(.*)', column_name, re.IGNORECASE)
                if match:
                    number = match.group(1).strip()
                    prediction_word = match.group(2).strip()
                    prediction_type = 'probability distribution'
                else:
                    number = None
                    prediction_word = column_name
                    prediction_type = 'probability distribution'
            else:
                prediction_word = column_name  # Default
                prediction_type = 'unknown'
                number = None
            return prediction_type, number, prediction_word

        def extract_metadata(self):
            metadata_dict = {}
            if self.metadata_column:
                # Parse metadata to get key-value pairs
                metadata_content = self.data[self.metadata_column].dropna().tolist()
                for item in metadata_content:
                    lines = item.strip().split('\n')
                    for line in lines:
                        if line.lower() == 'metadata':
                            continue
                        if ':' in line or '=' in line:
                            if ':' in line:
                                key, value = line.split(':', 1)
                            else:
                                key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            metadata_dict[key] = value
            else:
                print("No metadata column found.")
            return metadata_dict

        class TimeHandling:
            def __init__(self, data_preparation_instance, time_column=None):
                self.data_preparation = data_preparation_instance
                self.data = self.data_preparation.data
                self.time_column = time_column or self.data_preparation.time_column or 'Time'
                self.parsed_time_column = 'Parsed_Time'
                self.format_string = None  # To store custom format string if needed

            def custom_time_type(self, format_string):
                """
                Store custom time format string for parsing.
                """
                print(f"Setting custom time format string: {format_string}")
                self.format_string = format_string

        class TextPreparation:
            def __init__(self):
                print("Initializing TextPreparation...")

            def truncate_text(self, text, max_tokens=48):
                tokens = text.split()
                truncated_tokens = tokens[:max_tokens]
                truncated_text = ' '.join(truncated_tokens)
                if len(tokens) > max_tokens:
                    truncated_text += '...'
                return truncated_text

            def truncate_text_to_tokens(self, text, max_tokens=16):
                tokens = text.split()
                truncated_tokens = tokens[:max_tokens]
                truncated_text = ' '.join(truncated_tokens)
                if len(tokens) > max_tokens:
                    truncated_text += '...'
                return truncated_text

            def is_vowel(self, char):
                return char.lower() in ['a', 'e', 'i', 'o', 'u']

            def add_breaks_every_two_sentences(self, text):
                sentences = re.split(r'(?<=[.!?]) +', text)
                grouped_sentences = [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
                new_text = '<br>'.join(grouped_sentences)
                return new_text

    def create_metadata_info(self):
        sentences = []
        if self.your_dataset:
            sentences.append(f"This data was generated from {self.your_dataset}.")
        if self.our_dataset and self.background_dataset_size:
            sentences.append(f"{self.background_dataset_size} randomly selected data points were sampled from {self.our_dataset}.")
        elif self.our_dataset:
            sentences.append(f"Data points were sampled from {self.our_dataset}.")
        metadata_info = ' '.join(sentences)
        return metadata_info

    def get_color_sequence(self):
        color_sequence = [
            self.colors['forest_green'],      # Forest Green
            self.colors['burnt_orange'],      # Burnt Orange (darkened)
            self.colors['peach'],             # Peach (muted)
            self.colors['mustard_yellow'],    # Mustard Yellow (darkened)
            self.colors['dark_grey'],         # Dark Grey (slightly darker)
            self.colors['navy_blue'],         # Navy Blue (darkened)
            self.colors['chocolate_brown'],   # Chocolate Brown (darkened)
            self.colors['berry_red'],         # Berry Red (darkened)
            self.colors['light_gray'],        # Light Gray (unchanged)
        ]
        return color_sequence

class ScientificTable(BasePlot):
    def __init__(self, data, column_name, num_entries=10, fig_counter=1, prediction_word=None, **kwargs):
        super().__init__(data, **kwargs)
        self.column_name = column_name
        self.num_entries = num_entries
        self.fig_counter = fig_counter
        self.prediction_word = prediction_word or self.data_preparation.extract_prediction_info(self.column_name)[2]

    def create_table(self):
        # Extract relevant columns
        columns_to_include = []
        if self.data_preparation.text_column:
            columns_to_include.append(self.data_preparation.text_column)
        if self.data_preparation.source_column:
            columns_to_include.append(self.data_preparation.source_column)
        if self.data_preparation.time_column:
            columns_to_include.append(self.data_preparation.time_column)
        columns_to_include.append(self.column_name)

        df = self.data[columns_to_include].copy()

        # Identify data rows (exclude header or non-data rows)
        if self.data_preparation.text_column:
            data_rows = df[self.data_preparation.text_column] != self.data_preparation.text_column  # Exclude header rows
            # Apply truncate_text only to data rows
            df.loc[data_rows, self.data_preparation.text_column] = df.loc[data_rows, self.data_preparation.text_column].apply(
                self.data_preparation.text_preparation.truncate_text
            )

        # Sort the data by the score column
        df_sorted = df.sort_values(by=self.column_name, ascending=False)

        # Create copies to avoid SettingWithCopyWarning
        top_entries = df_sorted.head(self.num_entries).copy()
        bottom_entries = df_sorted.tail(self.num_entries).copy()

        # Label the entries
        top_entries['Category'] = 'Top Scores'
        bottom_entries['Category'] = 'Lowest Scores'

        combined_df = pd.concat([top_entries, bottom_entries], ignore_index=True)

        # Prepare table data with subheadings and cell colors
        table_data = []
        cell_fill_colors = []
        cell_font_colors = []
        current_category = ''
        for _, row in combined_df.iterrows():
            if row['Category'] != current_category:
                # Insert subheading
                subheading_row = [''] * len(columns_to_include)
                subheading_row[0] = f"<b>{row['Category']}</b>"
                table_data.append(subheading_row)
                current_category = row['Category']
                # Apply forest green background and white text to both subheadings
                row_fill_color = [self.colors['forest_green']] * len(columns_to_include)
                row_font_color = ['white'] * len(columns_to_include)
                cell_fill_colors.append(row_fill_color)
                cell_font_colors.append(row_font_color)
            # Append data row
            data_row = [row[col] if col in row else '' for col in columns_to_include]
            table_data.append(data_row)
            # Append default colors for data row
            row_fill_color = ['white'] * len(columns_to_include)
            row_font_color = ['black'] * len(columns_to_include)
            cell_fill_colors.append(row_fill_color)
            cell_font_colors.append(row_font_color)

        # Transpose data and colors for Plotly
        table_data_transposed = list(map(list, zip(*table_data)))
        cell_fill_colors_transposed = list(map(list, zip(*cell_fill_colors)))
        cell_font_colors_transposed = list(map(list, zip(*cell_font_colors)))

        # Create the figure description
        fig_description = self.construct_table_description()

        fig_description = self.data_preparation.text_preparation.add_breaks_every_two_sentences(fig_description)

        # Create the table figure
        fig = go.Figure(data=[go.Table(
            columnwidth=[1]*len(columns_to_include),
            header=dict(
                values=[f"<b>{col}</b>" for col in columns_to_include],
                fill_color=self.colors['forest_green'],
                font=dict(color='white', family=self.font),
                align='left',
                height=40,
            ),
            cells=dict(
                values=table_data_transposed,
                fill_color=cell_fill_colors_transposed,
                font=dict(color=cell_font_colors_transposed, family=self.font),
                align='left',
                height=30,
            )
        )])

        # Update layout with adjusted margins
        fig.update_layout(
            margin=dict(l=50, r=50, t=100, b=150),
            height=800,
        )

        # Add background shapes for title and figure description
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=1.0,
            x1=1,
            y1=1.08,
            fillcolor=self.colors['forest_green'],
            line=dict(width=0),
            layer='below',
        )

        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1,
            y1=-0.12,
            fillcolor=self.colors['forest_green'],
            line=dict(width=0),
            layer='below',
        )

        # Add title as an annotation at the top
        fig.add_annotation(
            text=self.construct_table_title(),
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.04,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(color='white', family=self.font, size=self.title_font_size),
            align='center',
            borderpad=0,
        )

        # Add the figure description as an annotation at the bottom
        fig.add_annotation(
            text=fig_description,
            xref='paper',
            yref='paper',
            x=0.5,
            y=-0.06,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(color='white', family=self.font),
            align='center',
            borderpad=0,
        )

        self.fig = fig

    def construct_table_title(self):
            title = f"<b>{self.prediction_word} Polarities: Highest and Lowest Predicted Scores Table </b>"
            return title

    def construct_table_description(self):
            fig_description = (
                f"Fig {self.fig_counter}: A table displaying the texts predicted to have the "
                f"{self.num_entries} highest and {self.num_entries} lowest {self.prediction_word} scores. "
                f"Higher scores indicate a greater likelihood the text contains characteristics lending themselves toward {self.prediction_word}."
            )
            metadata_info = self.create_metadata_info()
            if metadata_info:
                fig_description += f" {metadata_info}"
            return fig_description

    def display(self):
            self.fig.show()

    def save_png(self, filename):
            self.fig.write_image(filename)

    def construct_histogram_description(self):
        fig_description = (
            f"Fig {self.fig_counter}: A histogram displaying the distribution of predicted {self.sentiment_context} categories."
        )
        metadata_info = self.create_metadata_info()
        if metadata_info:
            fig_description += f" {metadata_info}"
        return fig_description

    def display(self):
        self.fig.show()

    def save_png(self, filename):
        self.fig.write_image(filename)

class HistogramPlot(BasePlot):

    def __init__(self, data, fig_counter=1, **kwargs):
        super().__init__(data, **kwargs)
        self.fig_counter = fig_counter

    def create_histogram(self):
        # Check if probability distribution columns are present
        prob_dist_cols = []
        for col in self.data_preparation.prediction_columns:
            prediction_type, number, prediction_word = self.data_preparation.extract_prediction_info(col)
            if prediction_type == 'probability distribution':
                prob_dist_cols.append((col, prediction_word))

        if not prob_dist_cols:
            print("No probability distribution columns found for histogram.")
            return  # No histogram to create

        # For each row, find the column with the highest value
        counts = {}
        for col, word in prob_dist_cols:
            counts[word] = 0

        for idx, row in self.data.iterrows():
            max_value = -np.inf
            max_word = None
            for col, word in prob_dist_cols:
                value = row[col]
                if pd.notnull(value) and value > max_value:
                    max_value = value
                    max_word = word
            if max_word:
                counts[max_word] += 1

        # Prepare data for histogram
        words = list(counts.keys())
        frequencies = [counts[word] for word in words]

        # Sort words and frequencies based on frequencies
        sorted_pairs = sorted(zip(frequencies, words), reverse=True)
        frequencies, words = zip(*sorted_pairs)

        # Create the histogram
        fig = go.Figure()

        # Use colors from the palette, skip forest_green
        color_sequence = self.get_color_sequence()
        bar_colors = color_sequence[1:len(words)+1]

        fig.add_trace(go.Bar(
            x=words,
            y=frequencies,
            marker_color=bar_colors,
        ))

        # Calculate y_max as the nearest multiple of 20 above the max bin frequency
        max_freq = max(frequencies)
        y_max = ((max_freq + 19) // 20) * 20  # Nearest multiple of 20 above max_freq

        # Calculate tick values at 20%, 40%, 60%, 80%, and 100% of y_max
        tickvals = [int(y_max * 0.2 * i) for i in range(1, 6)]

        # Update layout
        fig.update_layout(
            title=self.construct_histogram_title(),
            xaxis=self.construct_histogram_xaxis(),
            yaxis=self.construct_histogram_yaxis(y_max, tickvals),
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=100, b=150),
        )

        # Add figure description
        fig_description = self.construct_histogram_description()
        fig_description = self.data_preparation.text_preparation.add_breaks_every_two_sentences(fig_description)

        # Adjust the position of the figure description
        fig.add_annotation(
            text=fig_description,
            xref='paper',
            yref='paper',
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            showarrow=False,
            font=dict(color='black', family=self.font),
            align='center',
            borderpad=0,
        )

        # Add grid lines
        fig.update_xaxes(showgrid=True, gridcolor=self.colors['light_gray'])
        fig.update_yaxes(showgrid=True, gridcolor=self.colors['light_gray'])

        self.fig = fig

    def construct_histogram_title(self):
        if self.your_dataset and not self.our_dataset:
            title_text = f"{self.your_dataset} {self.sentiment_context} Histogram"
        elif not self.your_dataset and self.our_dataset:
            title_text = f"{self.our_dataset} {self.sentiment_context} Histogram"
        elif self.your_dataset and self.our_dataset:
            title_text = f"{self.sentiment_context} Histogram of {self.your_dataset} with Additional {self.our_dataset} as Background Data"
        else:
            title_text = f"{self.sentiment_context} Histogram"
        title = {
            'text': title_text,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(family=self.font, size=self.title_font_size, color=self.colors['forest_green'])
        }
        return title

    def construct_histogram_xaxis(self):
        xaxis = dict(
            title='',
            tickmode='array',
            tickvals=self.data.columns,
            ticktext=self.data.columns,
            showgrid=False,
            linecolor='black',
            tickfont=dict(family=self.font, size=self.axis_label_font_size, color=self.colors['forest_green']),
        )
        return xaxis

    def construct_histogram_yaxis(self, y_max, tickvals):
        yaxis = dict(
            title='Frequency',
            range=[0, y_max],
            tickmode='array',
            tickvals=tickvals,
            showgrid=True,
            gridcolor=self.colors['light_gray'],
            linecolor='black',
            tickfont=dict(family=self.font, size=self.tick_font_size, color=self.colors['forest_green']),
            titlefont=dict(color=self.colors['forest_green'], size=self.axis_title_font_size),
        )
        return yaxis

    def construct_histogram_description(self):
        fig_description = (
            f"Fig {self.fig_counter}: A histogram displaying the distribution of predicted {self.sentiment_context} categories."
        )
        metadata_info = self.create_metadata_info()
        if metadata_info:
            fig_description += f" {metadata_info}"
        return fig_description

    def display(self):
        self.fig.show()

    def save_png(self, filename):
        self.fig.write_image(filename)

class LineGraphPlot(BasePlot):
    def __init__(self, data, fig_counter=1, perplexity=30, learning_rate=200,
                 model_provider=None, model_name=None, fine_tune_version_name=None,
                 sentiment_context='', **kwargs):
        super().__init__(data, **kwargs)
        self.fig_counter = fig_counter
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.model_provider = model_provider
        self.model_name = model_name
        self.fine_tune_version_name = fine_tune_version_name
        self.sentiment_context = sentiment_context

    def create_line_graph(self):
        # Parse time data locally
        self.parse_time_data()

        # Proceed with the rest of the method as before
        parsed_time_column = 'Parsed_Time'
        if parsed_time_column not in self.data.columns:
            print("No parsed time data found.")
            return  # No time data to plot

        # Ensure 'Parsed_Time' has valid entries
        data_with_time = self.data.dropna(subset=[parsed_time_column])

        if data_with_time.empty:
            print("No valid time data after parsing.")
            return  # No valid time data

        # Update data used for plotting
        self.data = data_with_time
        self.data['Parsed_Time'] = self.data[parsed_time_column]

        # Decide which type of line graph to create
        # ... (rest of the method remains the same)

    def parse_time_data(self):
        """
        Parse time data specific to LineGraphPlot.
        """
        time_column = self.data_preparation.time_column or 'Time'
        format_string = self.data_preparation.time_handling.format_string

        def safe_parse(x):
            if pd.isnull(x):
                return pd.NaT
            try:
                if format_string:
                    return datetime.datetime.strptime(str(x), format_string)
                else:
                    return parse(str(x))
            except (ValueError, TypeError, OverflowError):
                return pd.NaT

        self.data['Parsed_Time'] = self.data[time_column].apply(safe_parse)

        # After parsing, check if any dates could not be parsed
        total_missing = self.data['Parsed_Time'].isna().sum()
        if total_missing > 0:
            print(f"Could not parse {total_missing} dates in '{time_column}' column.")

    class ContinuousLineGraph:
        def __init__(self, parent):
            self.parent = parent

        def create(self):
            data = self.parent.data
            continuous_cols = []
            for col in self.parent.data_preparation.prediction_columns:
                if 'continuous:' in col.lower():
                    continuous_cols.append(col)

            if not continuous_cols:
                print("No continuous prediction columns found.")
                return

            col = continuous_cols[0]
            # Extract only prediction_word since prediction_type and number are not used
            _, _, prediction_word = self.parent.data_preparation.extract_prediction_info(col)

            # Prepare data
            data = data.dropna(subset=[col])
            if data.empty:
                print("No data points with valid continuous predictions after dropping NaNs.")
                return
            sources = data[self.parent.data_preparation.source_column] if self.parent.data_preparation.source_column else pd.Series(['Unknown'] * len(data))
            unique_sources = sources.unique()

            # Call the method to handle appearance
            self.Continuous_Line_Appearance(data, col, prediction_word, sources, unique_sources)

        def Continuous_Line_Appearance(self, data, col, prediction_word, sources, unique_sources):
            # Prepare hover text
            if self.parent.data_preparation.text_column:
                data['Truncated_Text'] = data[self.parent.data_preparation.text_column].apply(
                    self.parent.data_preparation.text_preparation.truncate_text_to_tokens
                )
            else:
                data['Truncated_Text'] = ''

            # Prepare hover text including specified items
            def create_hover_text(row):
                hover_text = f"{self.parent.sentiment_context} essence score: {row[col]}<br>"
                hover_text += f"Date: {row['Parsed_Time'].strftime('%Y-%m-%d')}<br>"
                hover_text += f"Text: {row['Truncated_Text']}<br>"
                topic_number = row[self.parent.topic_column] if self.parent.topic_column else None
                topic_words = row[self.parent.topic_words_column] if self.parent.topic_words_column else None
                if self.parent.topic_column and topic_number == -1:
                    hover_text += "Text identified as a topic outlier."
                elif self.parent.topic_column and topic_number is not None and topic_words:
                    words_list = topic_words.strip('{}').split(', ')[:5]
                    top_words = ', '.join(words_list)
                    hover_text += f"Topic {int(topic_number)}: {top_words}"
                return hover_text

            data['Hover_Text'] = data.apply(create_hover_text, axis=1)

            # Prepare color mapping based on topics
            if self.parent.topic_column:
                topic_numbers = data[self.parent.topic_column].unique()
                color_sequence = self.parent.get_color_sequence()
                topic_color_map = {topic: color_sequence[i % len(color_sequence)] for i, topic in enumerate(topic_numbers)}
            else:
                topic_color_map = {}

            # Prepare color sequence for lines, skip forest_green
            color_sequence = self.parent.get_color_sequence()
            line_color_sequence = color_sequence[1:]

            # Plot lines for each source
            fig = go.Figure()

            for i, source in enumerate(unique_sources):
                source_data = data[sources == source]
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data[col],
                    mode='lines',
                    name=source,
                    line=dict(color=line_color_sequence[i % len(line_color_sequence)]),
                    hoverinfo='skip',
                ))
                # Add scatter points with topic-based colors
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data[col],
                    mode='markers',
                    name=source,
                    marker=dict(
                        size=5,
                        color=[topic_color_map.get(topic, 'grey') for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else self.parent.colors['forest_green'],
                        symbol=['circle' if topic != -1 else 'circle-open' for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else 'circle',
                        line=dict(width=2, color='white') if self.parent.topic_column else None,
                    ),
                    hovertext=source_data['Hover_Text'],
                    hoverinfo='text',
                    showlegend=False,
                ))

            # Update layout
            title = self.construct_line_graph_title('continuous', prediction_word)
            fig_description = self.construct_line_graph_description(prediction_word)

            # Add dataset and source information
            total_points = len(data)
            if self.parent.data_preparation.source_column:
                source_counts = data[self.parent.data_preparation.source_column].value_counts()
                source_info = ', '.join([f"{count} points from source {source}" for source, count in source_counts.items()])
                fig_description = f"{fig_description} This dataset contains {total_points} points. {source_info}"
            else:
                fig_description = f"{fig_description} This dataset contains {total_points} points."

            fig_description = self.parent.data_preparation.text_preparation.add_breaks_every_two_sentences(fig_description)

            # Adjust the position of the figure description
            fig.add_annotation(
                text=fig_description,
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                xanchor='center',
                yanchor='top',
                showarrow=False,
                font=dict(color='black', family=self.parent.font),
                align='center',
                borderpad=0,
            )

            # Assign fig to self.parent.fig before updating layout
            self.parent.fig = fig

            # Adjust legend to overlay on the plot
            fig.update_layout(
                title=self.construct_line_graph_layout_title(title),
                xaxis=self.construct_line_graph_xaxis(),
                yaxis=self.construct_line_graph_yaxis(prediction_word),
                legend=self.construct_line_graph_legend(),
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=100, b=150),
            )

        def construct_line_graph_title(self, graph_type, prediction_word=''):
            if graph_type == 'continuous':
                if self.parent.your_dataset and not self.parent.our_dataset:
                    title = f"The Time Evolution of {self.parent.sentiment_context} within {self.parent.your_dataset}"
                elif not self.parent.your_dataset and self.parent.our_dataset:
                    title = f"The Time Evolution of {self.parent.sentiment_context} within {self.parent.our_dataset}"
                elif self.parent.your_dataset and self.parent.our_dataset:
                    title = f"The Time Evolution of {self.parent.sentiment_context} within {self.parent.your_dataset}, with Background Data from {self.parent.our_dataset}"
                else:
                    title = f"The Time Evolution of {self.parent.sentiment_context}"
            else:
                title = ""
            return title

        def construct_line_graph_description(self, prediction_word='', additional_info=False):
            fig_description = (
                f"Fig {self.parent.fig_counter}: A line graph displaying the above text data's {self.parent.sentiment_context} evolving over time."
            )
            metadata_info = self.parent.create_metadata_info()
            if metadata_info:
                fig_description += f" {metadata_info}"
            return fig_description

        def construct_line_graph_layout_title(self, title_text):
            title = {
                'text': title_text,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family=self.parent.font, size=self.parent.title_font_size, color=self.parent.colors['forest_green'])
            }
            return title

        def construct_line_graph_xaxis(self):
            xaxis = dict(
                title='Time',
                showgrid=True,
                gridcolor=self.parent.colors['light_gray'],
                linecolor='black',
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            return xaxis

        def construct_line_graph_yaxis(self, prediction_word):
            yaxis = dict(
                title=prediction_word,
                showgrid=True,
                gridcolor=self.parent.colors['light_gray'],
                linecolor='black',
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            return yaxis

        def construct_line_graph_legend(self):
            legend = dict(
                font=dict(family=self.parent.font),
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='Black',
                borderwidth=1,
            )
            return legend

    class ExpectationLineGraph:
        def __init__(self, parent):
            self.parent = parent

        def create(self):
            data = self.parent.data
            # Prepare data
            prob_dist_cols = []
            prob_dist_words = []
            for col in self.parent.data_preparation.prediction_columns:
                prediction_type, number, prediction_word = self.parent.data_preparation.extract_prediction_info(col)
                if prediction_type == 'probability distribution':
                    prob_dist_cols.append(col)
                    prob_dist_words.append(prediction_word)

            if not prob_dist_cols:
                print("No probability distribution columns found for expectation line graph.")
                return

            # Build a list of the bin midpoints
            num_bins = len(prob_dist_cols)
            highest_number = num_bins
            bin_midpoints = []
            for i in range(1, num_bins + 1):
                midpoint = i * (1 / highest_number) - (1 / (2 * highest_number))
                bin_midpoints.append(midpoint)

            # Calculate expectation values
            expectation_values = []
            for idx, row in data.iterrows():
                probabilities = []
                for col in prob_dist_cols:
                    probabilities.append(row[col])
                probabilities = np.array(probabilities)
                if np.isnan(probabilities).any():
                    expectation_values.append(np.nan)
                    continue
                expectation = np.sum(probabilities * bin_midpoints)
                expectation_values.append(expectation)

            data['Expectation'] = expectation_values
            data = data.dropna(subset=['Expectation'])
            if data.empty:
                print("No data points with valid expectation values after dropping NaNs.")
                return

            # Call the method to handle appearance
            self.Expectation_Line_Appearance(data, prob_dist_cols, prob_dist_words, bin_midpoints)

        def Expectation_Line_Appearance(self, data, prob_dist_cols, prob_dist_words, bin_midpoints):
            # Prepare hover text
            if self.parent.data_preparation.text_column:
                data['Truncated_Text'] = data[self.parent.data_preparation.text_column].apply(
                    self.parent.data_preparation.text_preparation.truncate_text_to_tokens
                )
            else:
                data['Truncated_Text'] = ''

            # Prepare hover text including specified items
            def create_hover_text(row):
                hover_text = f"{self.parent.sentiment_context} essence score: {row['Expectation']}<br>"
                hover_text += f"Date: {row['Parsed_Time'].strftime('%Y-%m-%d')}<br>"
                hover_text += f"Text: {row['Truncated_Text']}<br>"
                topic_number = row[self.parent.topic_column] if self.parent.topic_column else None
                topic_words = row[self.parent.topic_words_column] if self.parent.topic_words_column else None
                if self.parent.topic_column and topic_number == -1:
                    hover_text += "Text identified as a topic outlier."
                elif self.parent.topic_column and topic_number is not None and topic_words:
                    words_list = topic_words.strip('{}').split(', ')[:5]
                    top_words = ', '.join(words_list)
                    hover_text += f"Topic {int(topic_number)}: {top_words}"
                return hover_text

            data['Hover_Text'] = data.apply(create_hover_text, axis=1)

            # Prepare color mapping based on topics
            if self.parent.topic_column:
                topic_numbers = data[self.parent.topic_column].unique()
                color_sequence = self.parent.get_color_sequence()
                topic_color_map = {topic: color_sequence[i % len(color_sequence)] for i, topic in enumerate(topic_numbers)}
            else:
                topic_color_map = {}

            # Prepare color sequence for lines, skip forest_green
            color_sequence = self.parent.get_color_sequence()
            line_color_sequence = color_sequence[1:]

            # Plot lines for each source
            sources = data[self.parent.data_preparation.source_column] if self.parent.data_preparation.source_column else pd.Series(['Unknown'] * len(data))
            unique_sources = sources.unique()

            fig = go.Figure()

            for i, source in enumerate(unique_sources):
                source_data = data[sources == source]
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data['Expectation'],
                    mode='lines',
                    name=source,
                    line=dict(color=line_color_sequence[i % len(line_color_sequence)]),
                    hoverinfo='skip',
                ))
                # Add scatter points with topic-based colors
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data['Expectation'],
                    mode='markers',
                    name=source,
                    marker=dict(
                        size=5,
                        color=[topic_color_map.get(topic, 'grey') for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else self.parent.colors['forest_green'],
                        symbol=['circle' if topic != -1 else 'circle-open' for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else 'circle',
                        line=dict(width=2, color='white') if self.parent.topic_column else None,
                    ),
                    hovertext=source_data['Hover_Text'],
                    hoverinfo='text',
                    showlegend=False,
                ))

            # Update layout
            title = self.construct_line_graph_title('expectation')
            fig_description = self.construct_line_graph_description(additional_info=True)

            # Add dataset and source information
            total_points = len(data)
            if self.parent.data_preparation.source_column:
                source_counts = data[self.parent.data_preparation.source_column].value_counts()
                source_info = ', '.join([f"{count} points from source {source}" for source, count in source_counts.items()])
                fig_description = f"{fig_description} This dataset contains {total_points} points. {source_info}"
            else:
                fig_description = f"{fig_description} This dataset contains {total_points} points."

            fig_description = self.parent.data_preparation.text_preparation.add_breaks_every_two_sentences(fig_description)

            # Adjust the position of the figure description
            fig.add_annotation(
                text=fig_description,
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                xanchor='center',
                yanchor='top',
                showarrow=False,
                font=dict(color='black', family=self.parent.font),
                align='center',
                borderpad=0,
            )

            # Assign fig to self.parent.fig before updating layout
            self.parent.fig = fig

            # Prepare y-axis ticks and grid lines
            data_min = data['Expectation'].min()
            data_max = data['Expectation'].max()

            # Adjust y-axis range if only one bin midpoint is within data range
            bin_midpoints_sorted = sorted(bin_midpoints)
            bin_midpoints_in_range = [bm for bm in bin_midpoints_sorted if data_min <= bm <= data_max]

            if len(bin_midpoints_in_range) < 3:
                data_mean = data['Expectation'].mean()
                closest_bin_index = np.argmin([abs(bm - data_mean) for bm in bin_midpoints_sorted])
                lower_index = max(closest_bin_index - 1, 0)
                upper_index = min(closest_bin_index + 1, len(bin_midpoints_sorted) - 1)
                y_min = bin_midpoints_sorted[lower_index]
                y_max = bin_midpoints_sorted[upper_index]
            else:
                y_min = data_min
                y_max = data_max

            # Expand y_min and y_max slightly for better visualization
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range

            # Ensure y_min and y_max are within [0,1]
            y_min = max(0, y_min)
            y_max = min(1, y_max)

            # Calculate grid lines every 20% of the y-range
            num_grid_lines = 5
            grid_line_positions = np.linspace(y_min, y_max, num_grid_lines + 1)

            # Prepare tickvals and ticktext
            tickvals = []
            ticktext = []

            # Add bin midpoints and words
            for bm, word in zip(bin_midpoints, prob_dist_words):
                if y_min <= bm <= y_max:
                    tickvals.append(bm)
                    ticktext.append(word)

            # Add grid line positions
            for pos in grid_line_positions:
                pos = round(pos, 5)
                if pos not in tickvals:
                    tickvals.append(pos)
                    ticktext.append(f"{pos:.2f}")

            # Sort tickvals and ticktext based on tickvals
            tick_pairs = sorted(zip(tickvals, ticktext))
            tickvals, ticktext = zip(*tick_pairs)

            # Update layout
            fig.update_layout(
                title=self.construct_line_graph_layout_title(title),
                xaxis=self.construct_line_graph_xaxis(),
                yaxis=self.construct_expectation_line_graph_yaxis(y_min, y_max, tickvals, ticktext, grid_line_positions),
                legend=self.construct_line_graph_legend(),
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=100, b=150),
            )

        # The following methods are specific to ExpectationLineGraph
        def construct_line_graph_title(self, graph_type):
            if graph_type == 'expectation':
                if self.parent.your_dataset and not self.parent.our_dataset:
                    title = f"Predicted {self.parent.sentiment_context} Score for {self.parent.your_dataset}"
                elif not self.parent.your_dataset and self.parent.our_dataset:
                    title = f"Predicted {self.parent.sentiment_context} Score for {self.parent.our_dataset}"
                elif self.parent.your_dataset and self.parent.our_dataset:
                    title = f"Predicted {self.parent.sentiment_context} Score for {self.parent.your_dataset} Alongside Additional Background Data from {self.parent.our_dataset}"
                else:
                    title = f"The Time Evolution of {self.parent.sentiment_context}"
            else:
                title = ""
            return title

        def construct_line_graph_description(self, additional_info=False):
            fig_description = (
                f"Fig {self.parent.fig_counter}: A line graph displaying the above text data's {self.parent.sentiment_context} evolving over time."
            )
            if additional_info:
                fig_description += " Additional technical information: The expectation value method assumes uniform distances between adjacent constructs and positions the mean at the centroid of the construct with the highest probability of membership."
            metadata_info = self.parent.create_metadata_info()
            if metadata_info:
                fig_description += f" {metadata_info}"
            return fig_description

        def construct_line_graph_layout_title(self, title_text):
            title = {
                'text': title_text,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family=self.parent.font, size=self.parent.title_font_size, color=self.parent.colors['forest_green'])
            }
            return title

        def construct_line_graph_xaxis(self):
            xaxis = dict(
                title='Time',
                showgrid=True,
                gridcolor=self.parent.colors['light_gray'],
                linecolor='black',
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            return xaxis

        def construct_expectation_line_graph_yaxis(self, y_min, y_max, tickvals, ticktext, grid_line_positions):
            yaxis = dict(
                title='Expectation Value',
                range=[y_min, y_max],
                showgrid=False,
                linecolor='black',
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            # Add grid lines manually
            for pos in grid_line_positions:
                self.parent.fig.add_shape(
                    type='line',
                    xref='paper',
                    x0=0,
                    x1=1,
                    yref='y',
                    y0=pos,
                    y1=pos,
                    line=dict(color=self.parent.colors['light_gray'], width=1),
                    layer='below',
                )
            return yaxis

        def construct_line_graph_legend(self):
            legend = dict(
                font=dict(family=self.parent.font),
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='Black',
                borderwidth=1,
            )
            return legend

    class UMAPLineGraph:
        def __init__(self, parent):
            self.parent = parent

        def create(self):
            data = self.parent.data
            # Prepare data
            prob_dist_cols = []
            prob_dist_words = []
            for col in self.parent.data_preparation.prediction_columns:
                prediction_type, number, prediction_word = self.parent.data_preparation.extract_prediction_info(col)
                if prediction_type == 'probability distribution':
                    prob_dist_cols.append(col)
                    prob_dist_words.append(prediction_word)

            if not prob_dist_cols:
                print("No probability distribution columns found for UMAP line graph.")
                return

            # Extract probabilities
            probabilities = data[prob_dist_cols].values
            if np.isnan(probabilities).any():
                valid_indices = ~np.isnan(probabilities).any(axis=1)
                probabilities = probabilities[valid_indices]
                data = data.iloc[valid_indices]

            if probabilities.size == 0:
                print("No data points with valid probabilities after removing NaNs.")
                return  # No data to plot

            umap_reducer = umap.UMAP(
                n_neighbors=2,
                n_components=1,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            umap_results = umap_reducer.fit_transform(probabilities)
            data['UMAP'] = umap_results[:, 0]

            self.UMAP_Line_Appearance(data)

        def UMAP_Line_Appearance(self, data):
            # Prepare hover text
            if self.parent.data_preparation.text_column:
                data['Truncated_Text'] = data[self.parent.data_preparation.text_column].apply(
                    self.parent.data_preparation.text_preparation.truncate_text_to_tokens
                )
            else:
                data['Truncated_Text'] = ''

            # Prepare hover text including specified items
            def create_hover_text(row):
                hover_text = f"{self.parent.sentiment_context} essence score: {row['UMAP']}<br>"
                hover_text += f"Date: {row['Parsed_Time'].strftime('%Y-%m-%d')}<br>"
                hover_text += f"Text: {row['Truncated_Text']}<br>"
                topic_number = row[self.parent.topic_column] if self.parent.topic_column else None
                topic_words = row[self.parent.topic_words_column] if self.parent.topic_words_column else None
                if self.parent.topic_column and topic_number == -1:
                    hover_text += "Text identified as a topic outlier."
                elif self.parent.topic_column and topic_number is not None and topic_words:
                    words_list = topic_words.strip('{}').split(', ')[:5]
                    top_words = ', '.join(words_list)
                    hover_text += f"Topic {int(topic_number)}: {top_words}"
                return hover_text

            data['Hover_Text'] = data.apply(create_hover_text, axis=1)

            # Prepare color mapping based on topics
            if self.parent.topic_column:
                topic_numbers = data[self.parent.topic_column].unique()
                color_sequence = self.parent.get_color_sequence()
                topic_color_map = {topic: color_sequence[i % len(color_sequence)] for i, topic in enumerate(topic_numbers)}
            else:
                topic_color_map = {}

            # Prepare color sequence for lines, skip forest_green
            color_sequence = self.parent.get_color_sequence()
            line_color_sequence = color_sequence[1:]

            # Plot lines for each source
            sources = data[self.parent.data_preparation.source_column] if self.parent.data_preparation.source_column else pd.Series(['Unknown'] * len(data))
            unique_sources = sources.unique()

            fig = go.Figure()

            for i, source in enumerate(unique_sources):
                source_data = data[sources == source]
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data['UMAP'],
                    mode='lines',
                    name=source,
                    line=dict(color=line_color_sequence[i % len(line_color_sequence)]),
                    hoverinfo='skip',
                ))
                # Add scatter points with topic-based colors
                fig.add_trace(go.Scatter(
                    x=source_data['Parsed_Time'],
                    y=source_data['UMAP'],
                    mode='markers',
                    name=source,
                    marker=dict(
                        size=5,
                        color=[topic_color_map.get(topic, 'grey') for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else self.parent.colors['forest_green'],
                        symbol=['circle' if topic != -1 else 'circle-open' for topic in source_data[self.parent.topic_column]] if self.parent.topic_column else 'circle',
                        line=dict(width=2, color='white') if self.parent.topic_column else None,
                    ),
                    hovertext=source_data['Hover_Text'],
                    hoverinfo='text',
                    showlegend=False,
                ))

            # Update layout
            title = self.construct_line_graph_title('umap')
            fig_description = self.construct_line_graph_description(umap=True)

            # Add dataset and source information
            total_points = len(data)
            if self.parent.data_preparation.source_column:
                source_counts = data[self.parent.data_preparation.source_column].value_counts()
                source_info = ', '.join([f"{count} points from source {source}" for source, count in source_counts.items()])
                fig_description = f"{fig_description} This dataset contains {total_points} points. {source_info}"
            else:
                fig_description = f"{fig_description} This dataset contains {total_points} points."

            fig_description = self.parent.data_preparation.text_preparation.add_breaks_every_two_sentences(fig_description)

            # Adjust the position of the figure description
            fig.add_annotation(
                text=fig_description,
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.175,
                xanchor='center',
                yanchor='top',
                showarrow=False,
                font=dict(color='black', family=self.parent.font),
                align='center',
                borderpad=0,
            )

            # Assign fig to self.parent.fig before updating layout
            self.parent.fig = fig

            # Adjust legend to overlay on the plot
            fig.update_layout(
                title=self.construct_line_graph_layout_title(title),
                xaxis=self.construct_line_graph_xaxis(),
                yaxis=self.construct_umap_line_graph_yaxis(),
                legend=self.construct_line_graph_legend(),
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=100, b=150),
            )

        def construct_line_graph_title(self, graph_type):
            if graph_type == 'umap':
                article = 'an' if self.parent.data_preparation.text_preparation.is_vowel(self.parent.sentiment_context[0]) else 'a'
                if self.parent.your_dataset and not self.parent.our_dataset:
                    title = f"The {self.parent.sentiment_context} UMAP Silhouette of {self.parent.your_dataset}"
                elif not self.parent.your_dataset and self.parent.our_dataset:
                    title = f"The {self.parent.sentiment_context} UMAP Silhouette of {self.parent.our_dataset}"
                elif self.parent.your_dataset and self.parent.our_dataset:
                    title = f"The {self.parent.sentiment_context} UMAP Silhouette of {self.parent.your_dataset} Alongside Additional Background {self.parent.our_dataset} Data"
                else:
                    title = f"A Visual Overview of {article} {self.parent.sentiment_context} Dimension Evolving over Time"
            else:
                title = ""
            return title

        def construct_line_graph_description(self, umap=False):
            fig_description = (
                f"Fig {self.parent.fig_counter}: A line graph displaying the above text data's {self.parent.sentiment_context} evolving over time."
            )
            if umap:
                fig_description += f" Additional technical information: The UMAP parameters used to generate this plot were n_neighbors=2, n_components=1, min_dist=0.0, metric='cosine', random_state=42."
            metadata_info = self.parent.create_metadata_info()
            if metadata_info:
                fig_description += f" {metadata_info}"
            return fig_description

        def construct_line_graph_layout_title(self, title_text):
            title = {
                'text': title_text,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(family=self.parent.font, size=self.parent.title_font_size, color=self.parent.colors['forest_green'])
            }
            return title

        def construct_line_graph_xaxis(self):
            xaxis = dict(
                title='Time',
                showgrid=True,
                gridcolor=self.parent.colors['light_gray'],
                linecolor='black',
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            return xaxis

        def construct_umap_line_graph_yaxis(self):
            yaxis = dict(
                title='Dimension Silhouette',
                showgrid=True,
                gridcolor=self.parent.colors['light_gray'],
                linecolor='black',
                tickfont=dict(family=self.parent.font, size=self.parent.tick_font_size, color=self.parent.colors['forest_green']),
                titlefont=dict(color=self.parent.colors['forest_green'], size=self.parent.axis_title_font_size),
            )
            return yaxis

        def construct_line_graph_legend(self):
            legend = dict(
                font=dict(family=self.parent.font),
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='Black',
                borderwidth=1,
            )
            return legend

    def create_metadata_info(self):
        metadata_info = ""
        if self.model_provider:
            metadata_info += f"The model provider was {self.model_provider}."
        if self.model_name:
            metadata_info += f" The model used was {self.model_name}."
        if self.fine_tune_version_name:
            metadata_info += f" The fine-tune version was {self.fine_tune_version_name}."
        if self.sentiment_context:
            metadata_info += f" The sentiment context was '{self.sentiment_context}'."
        return metadata_info.strip()

    def display(self):
        self.fig.show()

    def save_png(self, filename):
        self.fig.write_image(filename)

class LDA_Visualisation(BasePlot):
    def __init__(self, data):
        super().__init__(data)
        self.fig = None  # Initialize self.fig to None

    def create_lda_visualisation(self):
        if 'LDA_PCA_Coordinates' not in self.data.columns or 'LDA_Topics' not in self.data.columns:
            print("No LDA data was found. Skipping LDA visualization.")
            return  # Exit the method gracefully

        # Filter out rows where 'LDA_PCA_Coordinates' has missing values
        self.data = self.data.dropna(subset=['LDA_PCA_Coordinates'])

        # Split the 'LDA_PCA_Coordinates' column into separate x, y, z coordinates
        coordinates = self.data['LDA_PCA_Coordinates'].apply(lambda x: list(map(float, x.split(','))))
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        z_coords = [coord[2] for coord in coordinates]

        lda_topics = self.data['LDA_Topics']
        
        # Ensure we have a text column to display
        if 'Text' in self.data.columns:
            # Truncate the text to the first 16 tokens
            hover_texts = self.data['Text'].apply(self.truncate_text_to_tokens)
        else:
            hover_texts = [''] * len(self.data)  # If no text column, leave hover text empty

        # Create 3D scatter plot
        fig = go.Figure()

        # Map unique topics to colors
        unique_topics = lda_topics.unique()
        color_sequence = self.get_color_sequence()[1:]  # Skip forest green for data points
        color_map = {topic: color_sequence[i % len(color_sequence)] for i, topic in enumerate(unique_topics)}

        # Plot each point and color by LDA_Topic
        for topic in unique_topics:
            topic_data = self.data[self.data['LDA_Topics'] == topic]
            topic_coords = coordinates[lda_topics == topic]
            hover_texts_topic = hover_texts[lda_topics == topic]

            fig.add_trace(go.Scatter3d(
                x=[coord[0] for coord in topic_coords],
                y=[coord[1] for coord in topic_coords],
                z=[coord[2] for coord in topic_coords],
                mode='markers',
                marker=dict(size=5, color=color_map[topic]),
                name=f"Topic {topic}",
                hovertext=hover_texts_topic,  # Assign hover text for each point
                hoverinfo='text'  # Enable hover text
            ))

        # Set axis labels and title using forest green color
        fig.update_layout(
            title="LDA Representation of Text Data",
            title_font=dict(color=self.colors['forest_green']),  # Title color set to forest green
            scene=dict(
                xaxis_title='PCA component 1',
                yaxis_title='PCA component 2',
                zaxis_title='PCA component 3',
                xaxis=dict(
                    tickmode='linear', 
                    tick0=0, 
                    dtick=0.5,
                    title_font=dict(color=self.colors['forest_green']),  # X-axis title color
                    tickfont=dict(color=self.colors['forest_green'])      # X-axis tick color
                ),
                yaxis=dict(
                    tickmode='linear', 
                    tick0=0, 
                    dtick=0.5,
                    title_font=dict(color=self.colors['forest_green']),  # Y-axis title color
                    tickfont=dict(color=self.colors['forest_green'])      # Y-axis tick color
                ),
                zaxis=dict(
                    tickmode='linear', 
                    tick0=0, 
                    dtick=0.5,
                    title_font=dict(color=self.colors['forest_green']),  # Z-axis title color
                    tickfont=dict(color=self.colors['forest_green'])      # Z-axis tick color
                ),
            ),
            legend_title="LDA Topics",
            legend=dict(
                x=0,  # Position on the right
                y=0.5,  # Position at the top
                bgcolor='rgba(255, 255, 255, 0.5)',  # Optional background color
                bordercolor='black',
                borderwidth=1
            ),
            annotations=[dict(
                text=self.add_line_breaks(
                    "This is a Latent Dirichlet Allocation generated plot using principal component analysis "
                    "to display the 3 principal component dimensions identified within the data. Each color "
                    "represents an identified topic. We recommend forming topic conclusions on our BERT models, "
                    "and treating this LDA model as a legacy feature."
                ),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.1, y=1, 
                xanchor='center', yanchor='top',
                font=dict(color='black', size=14),  # Change text color to black
            )]
        )

        self.fig = fig  # Assign the figure to self.fig

    def display(self):
        """Display the plot if available."""
        if self.fig:
            self.fig.show()
        else:
            print("No LDA visualization available to display.")

    def save_png(self, filename):
        """Save the plot as a PNG image if available."""
        if self.fig:
            self.fig.write_image(filename)
        else:
            print("No LDA visualization available to save.")

class BERTVisualization(BasePlot):
    def __init__(self, data, bertopic_processor, **kwargs):
        super().__init__(data, **kwargs)
        self.bertopic_processor = bertopic_processor

    def parse_time_data(self):
        """
        Parse time data specific to BERTVisualization.
        """
        time_column = self.data_preparation.time_column or 'Time'
        format_string = self.data_preparation.time_handling.format_string

        def safe_parse(x):
            if pd.isnull(x):
                return pd.NaT
            try:
                if format_string:
                    return datetime.datetime.strptime(str(x), format_string)
                else:
                    return parse(str(x))
            except (ValueError, TypeError, OverflowError):
                return pd.NaT

        self.data['Parsed_Time'] = self.data[time_column].apply(safe_parse)

        # After parsing, check if any dates could not be parsed
        total_missing = self.data['Parsed_Time'].isna().sum()
        if total_missing > 0:
            print(f"Could not parse {total_missing} dates in '{time_column}' column.")

    def parse_timestamps(self):
        """
        Parses timestamps using local parsing.
        Returns texts, topics, and timestamps with valid parsed dates.
        """
        # Parse time data locally
        self.parse_time_data()

        parsed_time_column = 'Parsed_Time'
        if parsed_time_column not in self.data.columns:
            print("No parsed time data found.")
            return None, None, None

        # Drop rows where 'Parsed_Time' is NaT
        valid_time_data = self.data.dropna(subset=[parsed_time_column])

        if valid_time_data.empty:
            print("No valid time data after parsing.")
            return None, None, None
        else:
            print(f"Number of data points with valid time: {len(valid_time_data)}")
            # Proceed with the valid data
            texts = valid_time_data[self.data_preparation.text_column].tolist()
            timestamps = valid_time_data[parsed_time_column].tolist()

            # Ensure 'BERTopic_Topic' column exists
            topics_column = self.data_preparation.identify_column('BERTopic_Topic')
            if topics_column:
                topics = valid_time_data[topics_column].tolist()
            else:
                print("BERTopic_Topic column not found in data.")
                topics = None
            return texts, topics, timestamps

    def visualize_topics(self, topics=None, top_n_topics=None, use_ctfidf=False,
                         custom_labels=False, title='<b>Intertopic Distance Map</b>',
                         width=650, height=650):
        """
        Visualize topics, their sizes, and their corresponding words.
        """
        if self.bertopic_processor.topic_model is None:
            raise ValueError("BERTopic model is not initialized.")
        fig = self.bertopic_processor.topic_model.visualize_topics(
            topics=topics,
            top_n_topics=top_n_topics,
            custom_labels=custom_labels,
            title=title,
            width=width,
            height=height
        )
        return fig

    def visualize_topics_over_time(self, top_n_topics=None, topics=None,
                                   normalize_frequency=False, custom_labels=False,
                                   title='<b>Topics over Time</b>', width=1250, height=450):
        """
        Visualize topics over time.
        """
        if self.bertopic_processor.topic_model is None:
            raise ValueError("BERTopic model is not initialized.")

        # Parse timestamps and get texts, topics, and timestamps
        texts, topics_list, timestamps = self.parse_timestamps()
        if texts is None or timestamps is None:
            print("Cannot generate topics over time due to missing timestamps or texts.")
            return None

        # If topics_list is None, fit the model to get topics
        if topics_list is None:
            topics_list, _ = self.bertopic_processor.topic_model.fit_transform(texts)

        # Prepare topics over time
        topics_over_time = self.bertopic_processor.topic_model.topics_over_time(
            texts, timestamps, topics=topics_list
        )

        fig = self.bertopic_processor.topic_model.visualize_topics_over_time(
            topics_over_time=topics_over_time,
            top_n_topics=top_n_topics,
            topics=topics,
            normalize_frequency=normalize_frequency,
            custom_labels=custom_labels,
            title=title,
            width=width,
            height=height
        )
        return fig

    def visualize_barchart(self, topics=None, top_n_topics=48, n_words=10,
                           custom_labels=False, title='<b>Topic Word Scores</b>',
                           width=250, height=250):
        """
        Visualize a barchart of selected topics.
        """
        if self.bertopic_processor.topic_model is None:
            raise ValueError("BERTopic model is not initialized.")
        fig = self.bertopic_processor.topic_model.visualize_barchart(
            topics=topics,
            top_n_topics=top_n_topics,
            n_words=n_words,
            custom_labels=custom_labels,
            title=title,
            width=width,
            height=height,
        )
        return fig

    def save_visualize_topics_as_json(self, output_file="topics_visualization.json"):
        """
        Saves the BERTopic topic visualization as a JSON file.
        """
        if self.bertopic_processor.topic_model is None:
            raise ValueError("Topic model is not initialized. Please initialize and train the model first.")
        fig = self.bertopic_processor.topic_model.visualize_topics()
        fig_json = to_json(fig)

        with open(output_file, 'w') as json_file:
            json_file.write(fig_json)
        print(f"Topic visualization saved as {output_file}.")

    def save_visualize_heatmap_as_json(self, output_file="heatmap_visualization.json"):
        """
        Saves the BERTopic heatmap visualization as a JSON file.
        """
        if self.bertopic_processor.topic_model is None:
            raise ValueError("Topic model is not initialized. Please initialize and train the model first.")
        
        print("Generating heatmap visualization as JSON...")
        fig = self.bertopic_processor.topic_model.visualize_heatmap()
        fig_json = to_json(fig)

        with open(output_file, 'w') as json_file:
            json_file.write(fig_json)
        print(f"Heatmap visualization saved as {output_file}.")