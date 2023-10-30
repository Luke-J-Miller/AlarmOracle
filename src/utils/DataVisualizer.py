import matplotlib.pyplot as plt
import numpy as np

class Visualize:
    def __init__(self):
        pass

    def visualize(self, datasets_dict):
        for i, dataset_key in enumerate(datasets_dict.keys()):
            predicted_matrix = datasets_dict[dataset_key].get('predicted_matrix', None)
            ground_truth_matrix = datasets_dict[dataset_key].get('alarm_adj_matrix', None)
            masked_ground_truth_matrix = datasets_dict[dataset_key].get('masked_alarm_adj_matrix', None)

            if predicted_matrix is None or ground_truth_matrix is None or masked_ground_truth_matrix is None:
                print(f"Missing matrices for {dataset_key}. Skipping visualization.")
                continue

            confusion_matrix = np.zeros_like(predicted_matrix, dtype=int)

            tp_masked = np.logical_and(predicted_matrix == 1, masked_ground_truth_matrix == 1)
            tp_unmasked = np.logical_and(np.logical_and(predicted_matrix == 1, ground_truth_matrix == 1), masked_ground_truth_matrix == -1)
            fp = np.logical_and(predicted_matrix == 1, ground_truth_matrix == 0)
            confusion_matrix[tp_masked] = 1  # TP from masked
            confusion_matrix[tp_unmasked] = 2  # TP but not in masked
            confusion_matrix[fp] = 3  # FP

            plt.figure(figsize=(10, 10))

            # Define custom colormap
            cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'green', 'red'])

            plt.imshow(confusion_matrix, cmap=cmap, vmin=0, vmax=3)  # Setting vmin and vmax here
            #plt.colorbar(ticks=[0, 1, 2, 3], orientation='vertical')

            for (j, i), value in np.ndenumerate(confusion_matrix):
                plt.text(i, j, f"{value}", ha='center', va='center')

            plt.title(f'Confusion Matrix for {dataset_key}')

            # Create a legend
            import matplotlib.patches as mpatches
            legend_labels = {
                'white': 'Negative',
                'gray': 'Prior Info TP',
                'green': 'Predicted TP',
                'red': 'False Positive'
            }
            patches = [mpatches.Patch(color=color, label=label) for color, label in legend_labels.items()]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            plt.show()
