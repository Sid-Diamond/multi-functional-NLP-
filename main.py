import os
from src.dataset_handling import DatasetHandler

from src.alb_s1 import (
    SentimentFineTuner,
    SentimentInferencer,
    SentimentCSVDataSaver,
)

from src.lda_module import (
    LDATextProcessor,
    LDAProcessor,
    LDACSVDataSaver,
)

from src.kcluster_module import (
    TextProcessingAndDatabase,
    KclusterAnalysis,
    KclusterCSVDataSaver,       # ← now imported
)

from src.bertopic_module import (
    BERTopicProcessor,
    BERTopicCSVDataSaver,
    TextProcessing,
)

from src.metadata_module import MetadataCSVDataSaver

from src.Data_Visualisation import (
    BasePlot,
    ScientificTable,
    HistogramPlot,
    LineGraphPlot,
    LDA_Visualisation,         
    BERT_Visualisation,          
)

our_datasets = 'climatebert/climate_sentiment'
your_dataset_toggle = False
our_datasets_toggle = True

sample_size = 50
shuffle_data = True
split = 'train'

sentiment_analysis = True
sentiment_context = 'FAdam 14. Political Idealogy'
sentiment_mode = 'classic'
max_seq_length = 64
model_size = 'albert-base'

fine_tune_or_inference = 'inference'
start_from_context = False
USE_LOCAL_EXPERIMENTS = False
save_fine_tune = 'yes'

class_labels = ["Liberalism","Conservatism"]
societally_linear = "yes"
fine_tune_version_name = 'Fun Again hmmn'
fine_tune_quality = True
batch_size = 25
epochs = 10
lr = 1e-5
weight_decay = 0.01
betas = (0.9, 0.999)
clip = 1
p = 0.5
eps = 1e-8
diamond_uncertainty = True
consider_bias = 'false'
n_proportion = 0.1
f1_dropout_plot_step = (0.005,0.001,0.2)
starting_weight_f1_plot = 'true'
starting_weight_f1_points = 10


LDA_analysis = False
LDA_num_topics = 10

BERTopic_analysis = True
BERTopic_num_topics = 'auto'
bert_clean_internal = True
bert_clean_manual = True

Kcluster_analysis = False

Data_Visualisation = True 
display_in_browser = True
save_png = False
generate_table = True
num_table_entries = 10
generate_histogram = True
generate_line_graph = True
generate_LDA = True
generate_bertopic_visualizations = True
generate_bertopic_topics_visualization = True
generate_bertopic_dtm_visualization = True
generate_bertopic_barchart_visualization = True

# MAIN PIPELINE
MODEL_VERSIONS_ROOT = (
    "model_versions_local_experiments"
    if USE_LOCAL_EXPERIMENTS
    else "model_versions_git_ready"
)
# make available to downstream modules
os.environ["MODEL_VERSIONS_ROOT"] = MODEL_VERSIONS_ROOT

if __name__ == "__main__":
    dataset_handler = DatasetHandler(csv_file='output.csv')

    # 1) Possibly initialize CSV from external or huggingface dataset
    if not your_dataset_toggle and not our_datasets_toggle:
        print("Warning: Neither your_dataset_toggle nor our_datasets_toggle is True; no data loaded.")
    else:
        dataset_handler.initialize_csv(
            our_datasets=our_datasets if our_datasets_toggle else None,
            your_dataset_toggle=your_dataset_toggle,
            our_datasets_toggle=our_datasets_toggle,
            sample_size=sample_size,
            shuffle_data=shuffle_data,
            split=split
        )

    # 2) Sentiment Analysis: Fine-tuning or Inference
    if sentiment_analysis:
        if fine_tune_or_inference == 'fine_tune':
            fine_tuner = SentimentFineTuner(
                max_seq_length=max_seq_length,
                model_size=model_size
            )

            # Attach CSV saver
            fine_tuner.csv_saver = SentimentCSVDataSaver(
                dataset_handler=dataset_handler,
                sentiment_context=sentiment_context
            )

            # Prepare dataset
            fine_tune_dataset = {
                'train': fine_tuner.prepare_finetuning_dataset(
                    dataset_name=our_datasets,
                    split='train',
                    sample_size=sample_size,
                    shuffle_data=shuffle_data
                ),
                'test': fine_tuner.prepare_finetuning_dataset(
                    dataset_name=our_datasets,
                    split='test',
                    sample_size=sample_size,
                    shuffle_data=shuffle_data
                )
            }

            # Hyperparams
            hyperparams = {
                'lr': lr,
                'batch_size': batch_size,
                'epochs': epochs,
                'weight_decay': weight_decay,
                'betas': betas,
                'clip': clip,
                'p': p,
                'eps': eps
            }

            # Actually do the fine-tuning
            fine_tuner.fine_tune(
                dataset=fine_tune_dataset,
                sentiment_context=sentiment_context,
                start_from_context=start_from_context,
                save_fine_tune=save_fine_tune,
                fine_tune_version_name=fine_tune_version_name,
                fine_tune_quality=fine_tune_quality,
                hyperparams=hyperparams,
                n_proportion=n_proportion,
                class_labels=class_labels,
                societally_linear=societally_linear,
                starting_weight_f1_points=starting_weight_f1_points,
                starting_weight_f1_plot=starting_weight_f1_plot,
                diamond_uncertainty=diamond_uncertainty,      
                f1_dropout_plot_step=f1_dropout_plot_step     
            )

        else:
            # Inference scenario
            inferencer = SentimentInferencer(
                max_seq_length=max_seq_length,
                model_size=model_size
            )
            inferencer.initialize_model(sentiment_context=sentiment_context)
            texts_for_analysis = dataset_handler.get_texts_for_analysis()
            predictions = inferencer.run_inference(
                texts=texts_for_analysis,
                sentiment_context=sentiment_context,
                dataset_name=our_datasets if our_datasets_toggle else None,
                sentiment_mode=sentiment_mode
            )
            final_preds_only = []
            for (pred, _) in predictions:
                final_preds_only.append(pred)

            sentiment_csv_saver = SentimentCSVDataSaver(
                dataset_handler=dataset_handler,
                sentiment_context=sentiment_context
            )
            sentiment_csv_saver.save_inference_results(final_preds_only)

    # 3) LDA Analysis
    if LDA_analysis:
        lda_text_processor = LDATextProcessor()
        lda_processor = LDAProcessor(num_topics=LDA_num_topics)

        texts_for_analysis = dataset_handler.get_texts_for_analysis()
        lda_texts = lda_text_processor.preprocess_texts(texts_for_analysis)

        lda_model, corpus, dominant_topics = lda_processor.perform_lda(lda_texts)
        topic_matrix = lda_processor.get_topic_matrix(lda_model, corpus)
        pca_result = lda_processor.perform_pca(topic_matrix)
        tsne_result = lda_processor.perform_tsne(topic_matrix)

        lda_csv_saver = LDACSVDataSaver(dataset_handler)
        lda_csv_saver.save_results(
            pca_coordinates=pca_result.tolist(),
            tsne_coordinates=tsne_result.tolist(),
            dominant_topics=dominant_topics
        )

    # 4) KMeans Clustering Analysis
    if Kcluster_analysis:
        text_processor = TextProcessingAndDatabase(dataset_handler)
        kcluster_processor = KclusterAnalysis(n_components=3)

        df, texts_for_kcluster = text_processor.process_texts()
        if texts_for_kcluster:
            pca_results, labels = kcluster_processor.perform_analysis(texts_for_kcluster)
            kcluster_csv_saver = KclusterCSVDataSaver()
            kcluster_csv_saver.save_analysis_to_csv(
                pca_results, labels, dataset_handler
            )

    # 5) BERTopic Analysis
    if BERTopic_analysis:
        bertopic_processor = BERTopicProcessor(
            num_topics=BERTopic_num_topics,
            bert_clean_internal=bert_clean_internal
        )

        texts_for_bertopic = dataset_handler.get_texts_for_analysis()
        if bert_clean_manual:
            text_processor = TextProcessing()
            texts_for_bertopic = text_processor.preprocess_texts(texts_for_bertopic)
            print("Manual preprocessing applied for BERTopic texts.")

        bertopic_processor.initialize_model(len(texts_for_bertopic))
        bertopic_results = bertopic_processor.perform_bertopic(texts_for_bertopic)

        bertopic_csv_saver = BERTopicCSVDataSaver(dataset_handler, bertopic_processor)
        bertopic_csv_saver.save_results(bertopic_results)

    # 6) Metadata Analysis
    metadata_saver = MetadataCSVDataSaver(dataset_handler=dataset_handler)

    if fine_tune_or_inference == 'fine_tune':
        version_name = fine_tune_version_name
    else:
        version_name = sentiment_context

    if model_size == 'bert':
        model_type = 'bert-base-uncased'
    elif model_size == 'albert-base':
        model_type = 'albert-base-v2'
    elif model_size == 'xlarge':
        model_type = 'albert-xlarge-v2'
    else:
        model_type = 'unknown'

    metadata_saver.save_metadata(
        your_dataset_toggle=your_dataset_toggle,
        our_datasets=our_datasets,
        sample_size=sample_size,
        shuffle_data=shuffle_data,
        sentiment_context=sentiment_context if sentiment_analysis else None,
        LDA_analysis=LDA_analysis,
        LDA_num_topics=LDA_num_topics if LDA_analysis else None,
        version_name=version_name,
        model_type=model_type
    )

    # 7) Data Visualization
    if Data_Visualisation:
        data = dataset_handler.read_csv()
        fig_counter = 1

        base_plot = BasePlot(data=data, sentiment_context=sentiment_context)
        data_preparation = base_plot.data_preparation

        if generate_table and sentiment_analysis:
            prob_dist_cols = []
            for col in data_preparation.prediction_columns:
                ctype, num, cword = data_preparation.extract_prediction_info(col)
                if ctype == 'probability distribution':
                    prob_dist_cols.append(col)
            total_pdists = len(prob_dist_cols)

            for column in data_preparation.prediction_columns:
                ctype, num, cword = data_preparation.extract_prediction_info(column)
                if ctype == 'probability distribution' and num == '2' and total_pdists == 2:
                    continue

                table = ScientificTable(
                    data=data,
                    column_name=column,
                    num_entries=num_table_entries,
                    fig_counter=fig_counter,
                    prediction_word=cword,
                    sentiment_context=sentiment_context
                )
                table.create_table()
                if display_in_browser:
                    table.display()
                if save_png:
                    filename = f'table_{fig_counter}.png'
                    table.save_png(filename)
                fig_counter += 1

        if generate_histogram and (sentiment_analysis or LDA_analysis):
            histogram = HistogramPlot(
                data=data,
                fig_counter=fig_counter,
                title_font_size=24,
                axis_label_font_size=16,
                axis_title_font_size=20,
                tick_font_size=14,
                sentiment_context=sentiment_context
            )
            histogram.create_histogram()
            if hasattr(histogram, 'fig'):
                if display_in_browser:
                    histogram.display()
                if save_png:
                    filename = f'histogram_{fig_counter}.png'
                    histogram.save_png(filename)
                fig_counter += 1

        # The dataset handler stores timestamp information under the "Time"
        # column. The previous check looked for a literal 'time_field' column,
        # which does not exist and prevented line graphs from being generated.
        if generate_line_graph and ('Time' in data.columns):
            line_graph = LineGraphPlot(
                data=data,
                fig_counter=fig_counter,
                title_font_size=24,
                axis_label_font_size=16,
                axis_title_font_size=20,
                tick_font_size=14,
                sentiment_context=sentiment_context
            )
            line_graph.parse_time_data()
            line_graph.create_line_graph()
            if hasattr(line_graph, 'fig'):
                if display_in_browser:
                    line_graph.display()
                if save_png:
                    filename = f'line_graph_{fig_counter}.png'
                    line_graph.save_png(filename)
                fig_counter += 1
        elif generate_line_graph:
            print("Time data not found; skipping line graph generation.")

        if generate_LDA and LDA_analysis:
            lda_visualiser = LDA_Visualisation(data)
            lda_visualiser.create_lda_visualisation()
            lda_visualiser.display()
        elif generate_LDA:
            print("LDA analysis toggle is off; skipping LDA visualization.")

        if generate_bertopic_visualizations and BERTopic_analysis:
            if 'bertopic_processor' not in locals():
                print("No BERTopic processor found; skipping BERTopic visualizations.")
            else:
                bert_viz = BERT_Visualisation(
                    data=data,
                    bertopic_processor=bertopic_processor,
                    sentiment_context=sentiment_context
                )
                bert_viz.parse_time_data()

                if generate_bertopic_topics_visualization:
                    topics_fig = bert_viz.visualize_topics()
                    topics_fig.show()

                if generate_bertopic_dtm_visualization:
                    dtm_fig = bert_viz.visualize_topics_over_time()
                    if dtm_fig:
                        dtm_fig.show()
                    else:
                        print("Cannot generate topics-over-time: missing timestamps or topics.")

                if generate_bertopic_barchart_visualization:
                    bc_fig = bert_viz.visualize_barchart()
                    bc_fig.show()

    print("Main pipeline complete.")
