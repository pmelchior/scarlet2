import numpy as np
import scarlet
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


def get_extent(bbox):
    return [bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]]

def show_sources(
    scene,
    observation=None,
    norm=None,
    channel_map=None,
    figsize=None,
    model_mask=None,
    add_markers=True,
):

    sources = scene.sources
    n_sources = len(sources)

    panels = 4
    panel_size = 4.0
    skipped = 0
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False)

    marker_kwargs = {"mew": 1, "ms": 10}
    box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    for k, src in enumerate(sources):
        
        if hasattr(src.morphology.bbox, "center") and src.morphology.bbox.center is not None:
            center = np.array(src.morphology.bbox.center)[::-1]
        else:
            center = None
        
        start, stop = src.morphology.bbox.start[-2:][::-1], src.morphology.bbox.stop[-2:][::-1]
        points = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
        box_coords = [
            p for p in points
        ]
    
        # model in its bbox
        panel = 0
        model = src()

        # Show the unrendered model in it's bbox
        extent = get_extent(src.morphology.bbox)
        ax[k-skipped][panel].imshow(
            scarlet.display.img_to_rgb(model, norm=norm, channel_map=channel_map, mask=model_mask),
            extent=extent,
            origin="lower",
        )
        ax[k-skipped][panel].set_title("Model Source {}".format(k))
        if center is not None and add_markers:
            ax[k-skipped][panel].plot(*center, "wx", **marker_kwargs)
        panel += 1

        # model in observation frame
        model_ = observation.render(model)
        extent = get_extent(observation.frame.bbox)
        ax[k-skipped][panel].imshow(
            scarlet.display.img_to_rgb(model_, norm=norm, channel_map=channel_map),
            extent=extent,
            origin="lower",
        )
        ax[k-skipped][panel].set_title("Model Source {} Rendered".format(k))

        panel += 1

        # Center the observation on the source and display it
        _images = observation.data
        ax[k-skipped][panel].imshow(
            scarlet.display.img_to_rgb(_images, norm=norm, channel_map=channel_map),
            extent=extent,
            origin="lower",
        )
        ax[k-skipped][panel].set_title("Observation".format(k))
        if center is not None and add_markers:
            center_ = center
            ax[k-skipped][panel].plot(*center_, "wx", **marker_kwargs)
        poly = Polygon(box_coords, closed=True, **box_kwargs)
        ax[k-skipped][panel].add_artist(poly)
        panel += 1

        # needs to be evaluated in the source box to prevent truncation
        spectra = [src.spectrum()]
        
        for spectrum in spectra:
            ax[k-skipped][panel].plot(spectrum)
        ax[k-skipped][panel].set_xticks(range(len(spectrum)))
        if hasattr(observation.frame, "channels") and observation.frame.channels is not None:
            ax[k-skipped][panel].set_xticklabels(observation.frame.channels)
        ax[k-skipped][panel].set_title("Spectrum")
        ax[k-skipped][panel].set_xlabel("Channel")
        ax[k-skipped][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig