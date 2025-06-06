import logomaker
import matplotlib
import numpy
import pandas
from anndata import AnnData
from matplotlib.colors import ListedColormap


def basic_plot(adata: AnnData) -> int:
    """Generate a basic plot for an AnnData object."""
    print("Import matplotlib and implement a plotting function here.")
    return 0


class BasicClass:
    """A basic class."""

    my_attribute: str = "Some attribute."
    my_other_attribute: int = 0

    def __init__(self, adata: AnnData):
        print("Implement a class here.")

    def my_method(self, param: int) -> int:
        """A basic method."""
        print("Implement a method here.")
        return 0

    def my_other_method(self, param: str) -> str:
        """Another basic method."""
        print("Implement a method here.")
        return ""


def plot_seqlogo():
    """Plot a sequence logo using logomaker."""
    print("Implement a sequence logo plot here.")
    # Example usage of logomaker (uncomment when implemented)
    # df = pd.DataFrame(...)
    # logomaker.Logo(df)


# https://github.com/jmschrei/tangermeme/blob/main/tangermeme/plot.py
def check_box_overlap(box1, box2):
    """Check if annotation label text boxes overlap."""
    return not (box1.x0 >= box2.x1 or box2.x0 >= box1.x1 or box1.y0 >= box2.y1 or box2.y0 >= box1.y1)


def check_box_overlap_bar(box1, box2):
    """Check if annotation bars overlap."""
    return not (box1.x0 >= box2.x1 or box2.x0 >= box1.x1 or box1.y0 != box2.y0)


def place_new_box(box, box_list, n_tracks=4, show_extra=True):
    """Find a position for a new annotation label text box such that it does not overlap with previously plotted boxes."""
    box_height = box.y1 - box.y0
    box.y0 -= box_height
    box.y1 -= box_height

    if len(box_list) == 0:
        return box, 0

    overlap_exists = any(check_box_overlap(box, box2) for box2 in box_list)
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        if steps_down_taken == n_tracks:  # beyond the n_tracks limit: make boxes smaller
            if not show_extra:
                box.y0 -= box_height
                box.y1 -= box_height
                return box, steps_down_taken
            box.y1 -= box_height
            box.y0 -= box_height / 2
            box_height = box_height / 2
            overlap_exists = any(check_box_overlap(box, box2) for box2 in box_list)
            continue

        box.y0 -= box_height
        box.y1 -= box_height
        overlap_exists = any(check_box_overlap(box, box2) for box2 in box_list)

    return box, steps_down_taken


def place_new_bar(box, box_list, y_step=None, n_tracks=4, show_extra=True):
    """Find a position for a new annotation bar such that it does not overlap with previously plotted bars."""
    if y_step is None:
        raise ValueError("y_step must be provided.")

    if len(box_list) == 0:
        return box, 0

    overlap_exists = any(check_box_overlap_bar(box, box2) for box2 in box_list)
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        box.y0 -= y_step
        box.y1 -= y_step
        overlap_exists = any(check_box_overlap_bar(box, box2) for box2 in box_list)
    return box, steps_down_taken


def plot_logo(
    X_attr,
    ax=None,
    color=None,
    annotations=None,
    start=None,
    end=None,
    ylim=None,
    spacing=4,
    n_tracks=4,
    score_key="score",
    show_extra=True,
    show_score=True,
    annot_cmap="Set1",
):
    """
    Make a logo plot and optionally annotate it.

    Parameters
    ----------
    X_attr : array-like
        Attribution scores matrix for the sequence.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, the current axes will be used.
    color : str or tuple, optional
        Color for the logo plot.
    annotations : pandas.DataFrame, optional
        DataFrame containing annotations to be plotted.
    start : int, optional
        Start position for plotting a subset of the sequence.
    end : int, optional
        End position for plotting a subset of the sequence.
    ylim : float, optional
        Y-axis limits for the plot.
    spacing : int, default=4
        Spacing between annotations.
    n_tracks : int, default=4
        Number of annotation tracks to display.
    score_key : str, default="score"
        Column name in annotations containing the scores.
    show_extra : bool, default=True
        Whether to show extra annotations beyond n_tracks.
    show_score : bool, default=True
        Whether to show scores in annotation labels.
    annot_cmap : str or list or matplotlib.colors.ListedColormap, default="Set1"
        Colormap for annotations.

    Returns
    -------
    logo : logomaker.Logo
        The created logo object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("Must install matplotlib before using.") from err

    if start is not None and end is not None:
        X_attr = X_attr[:, start:end]

    if ax is None:
        ax = plt.gca()

    df = pandas.DataFrame(X_attr.T, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)

    if color is not None:
        alpha = numpy.array(["A", "C", "G", "T"])
        seq = "".join(alpha[numpy.abs(df.values).argmax(axis=1)])
        logo.style_glyphs_in_sequence(sequence=seq, color=color)

    # Set annotation colormap
    if isinstance(annot_cmap, str):
        cmap = plt.get_cmap(annot_cmap)
        # cmap = ListedColormap([(0,0,0)] + list(cmap.colors)) #add black as the first color
    elif isinstance(annot_cmap, list):
        cmap = ListedColormap(annot_cmap)
    elif isinstance(annot_cmap, matplotlib.colors.ListedColormap):
        cmap = annot_cmap

    if annotations is not None:
        start, end = start or 0, end or X_attr.shape[-1]

        annotations_ = annotations[annotations["start"] > start]
        annotations_ = annotations_[annotations_["end"] < end]
        annotations_ = annotations_.sort_values(["start"], ascending=True)
        # return annotations_

        ylim = ylim or max(abs(X_attr.min()), abs(X_attr.max()))
        ax.set_ylim(-ylim, ylim)
        # Remove unused variable r
        y_offset_bars = ax.get_ylim()[0] / 8
        # y_offset_labels = y_offset_bars +ax.get_ylim()[0]/10
        y_offset_labels = 0

        # deterrmine label text size and line width according to figure size
        bbox = ax.get_position()
        fig_width, fig_height = ax.get_figure().get_size_inches()
        width_in = bbox.width * fig_width
        # height_in variable is unused, removing it
        labelsize = width_in * 1.1
        linewidth = width_in * 0.25

        # plotting annotation labels
        label_box_objects = []
        visible_label_box_objects = []
        label_text_boxes = []
        text_box_colors = []
        for _, (_, row) in enumerate(annotations_.iterrows()):
            motif = row.values[0]
            motif_start = int(row["start"])
            motif_end = int(row["end"])
            score = row[score_key]
            motif_start -= start
            motif_end -= start

            # define label text
            text = str(motif)
            if show_score:
                text += f" ({score:3.3})"
            text = text.replace(" ", "\n")

            # plot text box in top most position...
            text_box = ax.text(motif_start, y_offset_labels, text, fontsize=labelsize)
            ax.get_figure().canvas.draw()
            bbox = text_box.get_window_extent()
            # ...shift box down if it overlaps with previously drawn boxes
            bbox_new, steps_down_taken = place_new_box(bbox, label_text_boxes, n_tracks=n_tracks, show_extra=show_extra)
            bbox_new_transformed = bbox_new.transformed(ax.transData.inverted())

            # color the text according to the number of downshifts
            text_color = cmap(steps_down_taken)
            if (
                steps_down_taken >= n_tracks
            ):  # if box is beyond the n_tracks limit, plot in smaller font size and in light grey.
                text_color = (0.7, 0.7, 0.7)
                text_box.set_fontsize(labelsize / 2)
                if not show_extra:
                    text_color = (1, 1, 1, 0)
            else:
                visible_label_box_objects.append(text_box)

            text_box.set_position((bbox_new_transformed.x0, bbox_new_transformed.y0))
            text_box.set_color(text_color)
            text_box_colors.append(text_color)
            label_text_boxes.append(bbox_new)
            label_box_objects.append(text_box)

        # plotting annotation bars
        bars_box_objects = []
        bars_boxes = []
        bars_ymins = []
        for idx, (_, row) in enumerate(annotations_.iterrows()):
            motif = row.values[0]
            motif_start = int(row["start"])
            motif_end = int(row["end"])
            score = row[score_key]
            motif_start -= start
            motif_end -= start

            bar_color = text_box_colors[idx]
            if bar_color == (0.7, 0.7, 0.7) or bar_color == (
                1,
                1,
                1,
                0,
            ):  # for labels beyond the n_tracks limit, no bar is plotted.
                continue

            xp = [motif_start, motif_end]
            yp = [y_offset_bars, y_offset_bars]

            # plot bar in topmost place...
            bar = ax.plot(xp, yp, color="0.3", linewidth=linewidth)
            ax.get_figure().canvas.draw()

            # ...shift bar down if it overlaps with previously drawn bars
            bar_box = bar[0].get_window_extent()
            bar_box_new, steps_down_taken = place_new_bar(
                bar_box, bars_boxes, y_step=linewidth * 2, n_tracks=n_tracks, show_extra=show_extra
            )
            bar_box_new_transformed = bar_box_new.transformed(ax.transData.inverted())
            bar[0].set_ydata([bar_box_new_transformed.y0, bar_box_new_transformed.y1])
            bars_boxes.append(bar_box_new)
            bar[0].set_color(bar_color)
            bars_ymins.append(bar_box_new_transformed.y0)
            bars_box_objects.append(bar[0])

        # shift text boxes down under the lowest bar
        bars_ymin = min(bars_ymins)
        for label_box in label_box_objects:
            label_box.set_y(label_box.get_position()[1] + bars_ymin)

        # if there is only one row of annotations, set colors to black
        if len({vis_box.get_color() for vis_box in visible_label_box_objects}) == 1:
            for label_box in visible_label_box_objects:
                label_box.set_color((0, 0, 0))
            for bar_box in bars_box_objects:
                bar_box.set_color((0, 0, 0))

    return logo
