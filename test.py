func(tables, func, *kwargs):
    
    output = pd.DataFrame
    gen = (obj for func(table) in tables.keys())
    for dataframe in gen:
        output.join(next(gen))
