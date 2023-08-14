from distutils.core import setup
import glob

setup(
    name="sentiment3d",
    packages=["sentiment3d"],
    description="COMPASS Pathways Three-dimesional Sentiment Model",
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
        "plotly",
        "kaleido",
    ],
    package_data={"sentiment3d": ["anchor_spec.json"]},
)
