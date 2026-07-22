"""
data_loading.py

Utility functions for loading intermediate datasets used throughout
the Parkinson's Disease Smartwatch Dataset (PADS) project.
"""

from pathlib import Path
import pandas as pd


def load_patients(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the patients dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to patients.csv.

    Returns
    -------
    pd.DataFrame
        Patient demographic and clinical information.
    """
    return pd.read_csv(csv_path)


def load_questionnaires(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the questionnaire dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to questionnaire.csv.

    Returns
    -------
    pd.DataFrame
        Questionnaire responses.
    """
    return pd.read_csv(csv_path)


def load_movement(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the movement metadata dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to movement.csv.

    Returns
    -------
    pd.DataFrame
        Movement metadata.
    """
    return pd.read_csv(csv_path)