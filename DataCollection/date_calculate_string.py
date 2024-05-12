import datetime
def dateadd(initial_date,date_add):
    start_date = datetime.datetime.strptime(initial_date, '%Y%m%d').date()
    delta = datetime.timedelta(days=date_add)
    new_date = start_date + delta
    return new_date.strftime('%Y%m%d')
def datesub(initial_date,date_sub):
    start_date = datetime.datetime.strptime(initial_date, '%Y%m%d').date()
    delta = datetime.timedelta(days=date_sub)
    new_date = start_date - delta
    return new_date.strftime('%Y%m%d')
def is_weekend(date_str):
    date_format = '%Y%m%d'
    date = datetime.datetime.strptime(date_str, date_format).date()
    # Check if the day of the week is Saturday (5) or Sunday (6)
    return date.weekday() == 5 or date.weekday() == 6