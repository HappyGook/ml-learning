import numpy as np
import plotly.graph_objects as go


def plot_gmm_3d(z, model, digit):

    assert z.shape[1] == 3, "Z must be 3D for 3D visualization"

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter3d(
        x=z[:, 0],
        y=z[:, 1],
        z=z[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.3),
        name="Data"
    ))

    # For each Gaussian
    for k in range(model.k):
        mean = model.means[k]
        cov = model.covariances[k]

        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)

        # Sort descending
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Create sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        sphere = np.stack([x, y, z], axis=-1)  # (30, 30, 3)

        # Scale by eigenvalues
        radii = 3 * np.sqrt(vals)
        ellipsoid = sphere @ np.diag(radii) @ vecs.T

        # Shift to mean
        ellipsoid += mean

        # Plot surface
        fig.add_trace(go.Surface(
            x=ellipsoid[..., 0],
            y=ellipsoid[..., 1],
            z=ellipsoid[..., 2],
            opacity=0.3,
            showscale=False,
            name=f"Gaussian {k}"
        ))

        # Plot mean
        fig.add_trace(go.Scatter3d(
            x=[mean[0]],
            y=[mean[1]],
            z=[mean[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name=f"Mean {k}"
        ))

    fig.update_layout(
        title=f"GMM clustering for digit {digit}",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()