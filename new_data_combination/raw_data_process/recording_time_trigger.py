from datetime import datetime, date, timedelta

def third_friday(year: int, month: int) -> date:
    """
    计算指定年月的第三个星期五日期。
    """
    # 本月第 1 天
    first_day = date(year, month, 1)
    # weekday(): 周一=0 … 周五=4 … 周日=6
    first_weekday = first_day.weekday()
    # 距离本月第一个星期五还需要几天
    days_until_first_friday = (4 - first_weekday + 7) % 7
    first_friday = first_day + timedelta(days=days_until_first_friday)
    # 第三个星期五 = 第一个星期五 + 2 周
    return first_friday + timedelta(weeks=2)

def second_friday(year: int, month: int) -> date:
    """
    计算指定年月的第三个星期五日期。
    """
    # 本月第 1 天
    first_day = date(year, month, 1)
    # weekday(): 周一=0 … 周五=4 … 周日=6
    first_weekday = first_day.weekday()
    # 距离本月第一个星期五还需要几天
    days_until_first_friday = (4 - first_weekday + 7) % 7
    first_friday = first_day + timedelta(days=days_until_first_friday)
    # 第2个星期五 = 第一个星期五 + 1 周
    return first_friday + timedelta(weeks=1)

def third_friday_trigger(now):
    expiry = third_friday(now.year, now.month)
    if now.date() > expiry:
        print(f"⚠️ 已经过了本月第三个星期五（{expiry}）")
        return 2
    elif now.date() == expiry:
        print(f"📌 今天就是本月第三个星期五（{expiry}）")
        return 1
    else:
        print(f"✅ 还没到本月第三个星期五（{expiry}）")
        return 0

def second_friday_trigger(now):
    expiry = second_friday(now.year, now.month)
    if now.date() > expiry:
        print(f"⚠️ 已经过了本月第二个星期五（{expiry}）")
        return 2
    elif now.date() == expiry:
        print(f"📌 今天就是本月第二个星期五（{expiry}）")
        return 1
    else:
        print(f"✅ 还没到本月第二个星期五（{expiry}）")
        return 0
    
def leap_month_trigger(now):
    if now.month in {3, 6, 9, 12}:
        print(f"📌 到期月到了")
        return 1
    else:
        print(f"✅ 非到期月")
        return 0

def is_trigger_day_on():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and third_friday_trigger(now)==1 :
        return 1
    else:
        return 0

def is_trigger_day_on_or_pass():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and third_friday_trigger(now)>0 :
        return 1
    else:
        return 0

def is_trigger_day_pass():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and third_friday_trigger(now)==2 :
        return 1
    else:
        return 0
    


#is_trigger_day_on()
#is_trigger_day_on_or_pass()
#is_trigger_day_pass()

def is_trigger_day2_on():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and second_friday_trigger(now)==1 :
        return 1
    else:
        return 0

def is_trigger_day2_on_or_pass():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and second_friday_trigger(now)>0 :
        return 1
    else:
        return 0

def is_trigger_day2_pass():
    now = datetime.now()
    print('现在是: ',now)
    if leap_month_trigger(now) and second_friday_trigger(now)==2 :
        return 1
    else:
        return 0