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
