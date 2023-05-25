from distutils.core import setup

setup(
    name="sentiment3d",
    packages=["sentiment3d"],
    description="COMPASS Pathways Three-dimesional Sentiment Model",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "sklearn",
        "patsy",
        "torch",
        "transformers",
        "codenamize",
        "plotly",
        "kaleido",
        "jupyterlab",
        "ipywidgets",
    ],
)
