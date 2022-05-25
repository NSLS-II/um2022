import torch.cuda

from constrainedmf.wrappers.scattering import iterative_decomposition, _decomposition_preprocess
from constrainedmf.nmf.models import NMF, NMFD
import numpy as np
from matplotlib.pyplot import figure
import matplotlib as mpl
from IPython import display

from collections import namedtuple

TransformPair = namedtuple("TransformPair", ["forward", "inverse"])


def default_transform_factory():
    """
    Constructs simple transform that does nothing.
    Forward goes from scientific coordinates to beamline coordinates
    Reverse goes from beamline coordinates to scientific coordinates
    """
    return TransformPair(lambda x: x, lambda x: x)


def min_max_normalize(x, axis=-1):
    """
    Min max normalization
    Parameters
    ----------
    x: array
    axis: int
    Returns
    -------
    """
    return (np.array(x) - np.min(x, axis=axis, keepdims=True)) / (
        np.max(x, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True)
    )


def waterfall(ax, x, ys, alphas=None, color="k", sampling=1, offset=0.2, **kwargs):
    """
    Waterfall plot on axis.

    Parameters
    ----------
    ax: Axes
    x: array
        1-d array for shared x value
    ys: array
        2-d array of y values to sample
    alphas: array, None
        1-d array of alpha values for each sample
    color
        mpl color
    sampling: int
        Sample rate for full ys set
    offset: float
        Offset to place in waterfall
    kwargs

    Returns
    -------

    """
    if alphas is None:
        alphas = np.ones_like(ys[:, 0])
    indicies = range(0, ys.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs)



def independent_waterfall(
    ax, independents, x, ys, alphas=None, color="k", sampling=1, offset=0.2, **kwargs
):
    """
    Waterfall plot on axis.

    Parameters
    ----------
    ax: Axes
    independents: array
        Collection of independent variables to label by
    x: array
        1-d array for shared x value
    ys: array
        2-d array of y values to sample
    alphas: array, None
        1-d array of alpha values for each sample
    color
        mpl color
    sampling: int
        Sample rate for full ys set
    offset: float
        Offset to place in waterfall
    kwargs

    Returns
    -------

    """
    if alphas is None:
        alphas = np.ones_like(ys[:, 0])
    indicies = range(0, ys.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs, label=independents[idx])
        ax.set_yticks(
            [
                np.min(ys[indicies[0], :]),
                np.min(ys[indicies[indicies[len(indicies) // 2]], :])
                + len(indicies) // 2 * offset,
                np.min(ys[indicies[-1], :]) + len(indicies) * offset,
            ]
        )
        ax.set_yticklabels(
            [
                independents[indicies[0]],
                independents[indicies[len(indicies) // 2]],
                independents[indicies[-1]],
            ]
        )


def refresh_figure(fig):
    """

    Parameters
    ----------
    fig: Figure

    Returns
    -------

    """
    fig.patch.set_facecolor("white")
    fig.set_tight_layout(True)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    # display.clear_output(wait=True)
    # display.display(fig)


def decomposition(
    Q,
    I,  # noqa: E741
    *,
    n_components=3,
    q_range=None,
    initial_components=None,
    fix_components=(),
    mode="Linear",
    kernel_width=1,
    max_iter=1000,
    bkg_removal=None,
    normalize=False,
    device=None,
    **kwargs,
):
    """
    Decompose and label a set of I(Q) data with optional focus bounds. Can be used for other
    1-dimensional response functions, written with total scattering in mind.

    Two operating modes are available: Linear (conventional) and Deconvolutional. The former will proceed as conventional
    NMF as implemented in sklearn, with the added flexibility of the torch implementation. The latter will include a
    convolutional kernel in the reconstruction between the component and weight matricies.

    Initial components can be set as starting conditions of the component matrix for the optimization. These components
    can be fixed or allowed to vary using the `fix_components` argument as a tuple of booleans.

    Keyword arguments are passed to the fit method

    Parameters
    ----------
    Q : array
        Ordinate Q for I(Q). Assumed to be rank 2, shape (m_patterns, n_data)
    I : array
        The intensity values for each Q, assumed to be the same shape as Q. (m_patterns, n_data)
    n_components: int
        Number of components for NMF
    q_range : tuple, list
        (Min, Max) Q values for consideration in NMF. This enables a focused region for decomposition.
    initial_components: array
        Initial starting conditions of intensity components. Assumed to be shape (n_components, n_data).
        If q_range is given, these will be trimmed in accordance with I.
    fix_components: tuple(bool)
        Flags for fixing a subset of initial components
    mode: {"Linear", "Deconvolutional"}
        Operating mode
    kernel_width: int
        Width of 1-dimensional convolutional kernel
    max_iter: int
        Maximum number of iterations for NMF
    bkg_removal: int, optional
        Integer degree for peakutils background removal
    normalize: bool, optional
        Flag for min-max normalization
    device: str, torch.device, None
            Device for matrix factorization to proceed on. Defaults to cpu.
    **kwargs: dict
        Arguments passed to the fit method. See nmf.models.NMFBase.

    Returns
    -------
    sub_Q : array
        Subsampled ordinate used for NMF
    sub_I : array
        Subsampled I used for NMF
    alphas : array
        Resultant weights from NMF
    components:  array
        Resultant components from NMF

    """

    sub_Q, sub_I, idx_min, idx_max = _decomposition_preprocess(
        Q=Q, I=I, q_range=q_range, bkg_removal=bkg_removal, normalize=normalize
    )

    # Initial components
    if mode != "Deconvolutional":
        kernel_width = 1
    n_features = sub_I.shape[1]
    if initial_components is None:
        input_H = None
    else:
        input_H = []
        for i in range(n_components):
            try:
                sub_H = initial_components[i][idx_min:idx_max]
                sub_H = sub_H[kernel_width // 2 : len(sub_H) - kernel_width // 2 + 1]
                if normalize:
                    sub_H = (sub_H - np.min(sub_H)) / (np.max(sub_H) - np.min(sub_H))
                input_H.append(
                    torch.tensor(sub_H, dtype=torch.float).reshape(
                        1, n_features - kernel_width + 1
                    )
                )
            except IndexError:
                input_H.append(torch.rand(1, n_features - kernel_width + 1))

    # Model construction
    if mode == "Linear":
        model = NMF(
            sub_I.shape,
            n_components,
            initial_components=input_H,
            fix_components=fix_components,
            device=device,
        )
    elif mode == "Deconvolutional":
        model = NMFD(
            sub_I.shape,
            n_components,
            T=kernel_width,
            initial_components=input_H,
            fix_components=fix_components,
            device=device,
        )
    else:
        raise NotImplementedError

    W = model.fit_transform(torch.tensor(sub_I), max_iter=max_iter, **kwargs)

    if len(W.shape) > 2:
        alphas = torch.mean(W, 2).data.numpy()
    else:
        alphas = W.data.numpy()

    components = torch.cat([x for x in model.H_list]).data.numpy()
    return sub_Q, sub_I, alphas, components


class NMFCompanion:
    def __init__(
        self,
        n_components,
        *,
        q,
        coordinate_transform=None,
        deconvolutional=False,
        kernel_width=None,
        fixed_components=(),
        normalize=True,
        fig=None,
        cmap="tab10",
        device=None,
    ):
        """
        Base class for NMF companion agent.
        Parameters
        ----------
        n_components: int
            Number of components for NMF
        q: array
            Q space for measurement
        coordinate_transform: Callable
            Optional transformation for independent variables in tell.
            Useful for converting "scientific" space coordinates to less interpretable or reduced
            "beamline" space coordinates.
        deconvolutional: bool
            Operational mode for NMF.
        kernel_width: int
            Width of 1-dimensional convolutional kernel, required if deconvolutional is True.
        fixed_components: None, array
            Initial fixed components for NMF decomposition
        normalize: bool
            Normalize data in decomposition
        fig: Figure
        cmap: str
            Matplotlib colormap


        Returns
        -------

        """
        self.n_components = n_components
        self.q = q
        self.independent = None
        self.dependent_components = None  # NMF Components
        self.dependent_weights = None  # NMF Weights
        self.dependent = None  # Raw Data
        if coordinate_transform is None:
            self.coordinate_transform = default_transform_factory()
        else:
            self.coordinate_transform = coordinate_transform
        self.deconvolutional = deconvolutional
        self.fixed_components = fixed_components
        self.normalize = normalize
        if fig is None:
            self.fig = None #figure()
        else:
            self.fig = fig
        #axes = self.fig.subplots(2, 2)
        #self.component_ax = axes[0, 0]
        #self.weight_ax = axes[0, 1]
        #self.loss_ax = axes[1, 0]
        #self.residual_ax = axes[1, 1]
        self.plot_order = list(range(n_components))  # Order for plotting
        self.cmap = mpl.cm.get_cmap(cmap)
        self.norm = mpl.colors.Normalize(vmin=0, vmax=n_components)

        if self.deconvolutional and kernel_width is None:
            raise ValueError(
                "kernel_width is a required argument for NMFCompanion when deconvolutional mode is used."
            )

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

    def update_decomposition(self):
        if self.deconvolutional:
            mode = "Deconvolutional"
        else:
            mode = "Linear"
        _, _, self.dependent_weights, self.dependent_components = decomposition(
            np.zeros_like(
                self.dependent
            ),  # This is normally for Q tracking but irrelevant for the whole data range.
            self.dependent,
            n_components=self.n_components,
            initial_components=self.fixed_components,
            fix_components=[True for _ in range(len(self.fixed_components))],
            mode=mode,
            device=self.device,
        )

    def tell(self, x, y):
        """
        Tell the NMF about something new
        Parameters
        ----------
        x: These are the interesting parameters
        y: This should be the I(Q) shape (1, n_datapoints)

        Returns
        -------
        """
        ys = np.reshape(y, (1, -1))
        xs = np.reshape(x, (1, -1))
        self.tell_many(xs, ys)

    def tell_many(self, xs, ys):
        """
        Tell the NMF about many new things
        Parameters
        ----------
        xs: These are the interesting parameters, they get converted to  space via a transform
        ys: list, arr
            This should be a list length m of the Q/I(Q) shape (n, 2)

        Returns
        -------

        """
        new_independents = list()
        for i in range(xs.shape[0]):
            new_independents.append(self.coordinate_transform.forward(*xs[i, :]))
        if self.normalize:
            new_dependents = min_max_normalize(np.array(ys))
        else:
            new_dependents = np.array(ys)
        if self.independent is None:
            self.independent = np.array(new_independents)
            self.dependent = new_dependents
        else:
            self.independent = np.vstack([self.independent, new_independents])
            self.dependent = np.vstack([self.dependent, new_dependents])

    def update_plot_order(self):
        """
        Order by proxy center of mass of class in plot regime.
        Makes the plots feel like a progression not random.
        """
        self.plot_order = np.argsort(np.argmax(self.dependent_weights, axis=0))

    def update_weights_plot(self):
        self.weight_ax.cla()
        for i in range(self.dependent_weights.shape[1]):
            self.weight_ax.plot(
                self.independent,
                self.dependent_weights[:, self.plot_order[i]],
                color=self.cmap(self.norm(i)),
                label=f"Component {i + 1}",
            )
        self.weight_ax.set_xlim([np.min(self.independent), np.max(self.independent)])
        self.weight_ax.set_xlabel("Independent Variable")
        self.weight_ax.set_ylabel("Weight")

    def update_loss_plot(self):
        self.loss_ax.cla()
        WH = np.matmul(self.dependent_weights, self.dependent_components)
        loss = np.mean((WH - self.dependent) ** 2, axis=1)
        self.loss_ax.plot(self.independent, loss)
        self.loss_ax.set_xlim([np.min(self.independent), np.max(self.independent)])
        self.loss_ax.set_xlabel("Independent Variable")
        self.loss_ax.set_ylabel("Relative Error")
        self.loss_ax.set_yticks([])

    def update_component_plot(self):
        self.component_ax.cla()
        kernel_width = len(self.q) - self.dependent_components.shape[1] + 1
        prev_max = 0
        for i in range(self.dependent_components.shape[0]):

            if kernel_width == 1:
                self.component_ax.plot(
                    self.q,
                    self.dependent_components[self.plot_order[i], :] + prev_max,
                    color=self.cmap(self.norm(i)),
                )
            else:
                start_idx = kernel_width // 2
                finish_index = -kernel_width // 2 + 1
                self.component_ax.plot(
                    self.q[start_idx:finish_index],
                    self.dependent_components[self.plot_order[i], :] + prev_max,
                    color=self.cmap(self.norm(i)),
                )
            prev_max += np.max(self.dependent_components[self.plot_order[i], :])
        self.component_ax.set_xlabel(r"Q [$\AA^{-1}$]")
        self.component_ax.set_xlabel(r"2$\theta$ [degrees]")
        self.component_ax.set_ylabel("Stacked Intensity [Arb.]")
        self.component_ax.set_yticks([])

    def update_residual_plot(self):
        self.residual_ax.cla()
        residuals = (
            np.matmul(self.dependent_weights, self.dependent_components)
            - self.dependent
        )
        alpha = min_max_normalize(np.mean(residuals ** 2, axis=1))
        independent_waterfall(
            self.residual_ax, self.independent, self.q, residuals, alphas=alpha
        )
        self.residual_ax.set_xlabel(r"Q [$\AA^{-1}$]")
        self.residual_ax.set_xlabel(r"2$\theta$ [degrees]")
        self.residual_ax.set_ylabel("Independent Var")

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError

    def report(self, **kwargs):
        """Allow the agent to summarize observations"""
        self.update_decomposition()
        #self.update_plot_order()
        #self.update_weights_plot()
        #self.update_component_plot()
        #self.update_loss_plot()
        #self.update_residual_plot()

        # Polish the rest off
        #refresh_figure(self.fig)

    def __len__(self):
        if self.dependent is None:
            return 0
        else:
            return self.dependent.shape[0]


class AutoNMFCompanion(NMFCompanion):
    def __init__(self, n_components, *, q, **kwargs):
        super().__init__(n_components, q=q, **kwargs)

    def update_decomposition(self):
        # This loop will protect against some instability causing NaN values deep in the NMF source
        while (self.dependent_weights is None) or np.any(
            np.isnan(self.dependent_weights)
        ):
            (
                _,
                _,
                self.dependent_weights,
                self.dependent_components,
            ) = iterative_decomposition(
                self.q[None, :],
                self.dependent,
                n_components=self.n_components,
                mode="Linear",
                normalize=self.normalize,
                device=self.device
            )