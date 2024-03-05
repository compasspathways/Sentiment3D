from distutils.core import setup
import glob

setup(
    name="sentiment3d",
    packages=["sentiment3d"],
    long_description_description="COMPASS Pathways Three-dimesional Sentiment Model",
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "patsy",
        "torch",
        "transformers",
        "codenamize",
    ],
    extras_requires={
        "explore": [
            "plotly",
            "kaleido",
            "jupyterlab",
            "ipywidgets",
        ],
        "dev": ["pre-commit"],
    },
    package_data={"sentiment3d": ["anchor_spec.json"]},
)
