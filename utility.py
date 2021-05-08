import pandas as pd


def check_X_domain_column(df: pd.DataFrame, domain_column: str) -> None:

    assert isinstance(domain_column, str)
    if domain_column not in df.columns:
        raise ValueError('Cannot find the domain column: {}'.format(domain_column))
    elif set(df['domain'].unique()) != {'target', 'source'}:
        raise ValueError('Have to encode the domain column as either target or source in strings')
    else:
        pass
