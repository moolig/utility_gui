import os
import pandas as pd
import pygsheets


st_yes = 'כן'
st_no = 'לא'
st_religes_school = 'ממלכתי דתי'
st_normal_school = 'ממלכתי'
st_more_from_3 = '3 ומעלה'
st_is_brother = 'כן, אני אח/ות של בעל מגרש ברמת טראמפ'
st_is_parent = 'כן, אני הורה של בעל מגרש ברמת טראמפ'
st_couple_no_child = 'זוג, ללא ילדים מתחת לגיל 18 המתגוררים במשק הבית'
st_yes_both = 'כן, שני בני הזוג'
st_yes_one = 'כן, אחד מבני הזוג / יחיד'
st_couple_no_child_new = 'זוג- נשוי או ידוע בציבור, שמנהל משק בית משותף ומתגורר באופן קבוע יחד באותו בית מגורים, ללא ילדים, ולפחות אחד מבני הזוג מתחת לגיל 35'



st_idx = 'index'
st_point = 'ניקוד מסכם'
st_Timestamp = 'Timestamp'
st_email = 'Email address'
st_famely_name = 'שם משפחה'
st_name1 = 'שם פרטי מועמד 1'
st_name2 = 'שם פרטי מועמד 2'
st_fon1 = 'מספר טלפון של מועמד 1'
st_fon2 = 'מספר טלפון של מועמד 2'
st_contact_mail = 'כתובת מייל מועדפת ליצירת קשר'
st_eat_coser = 'האם המטבח בביתכם כשר?'
st_prire = 'האם אתם שותפים באורח קבע בתפילות שחרית בשבתות בבית כנסת?'
st_drive_shbat = 'האם אתם נוסעים בשבת?'
st_school = 'לאיזה זרם חינוכי אתם שולחים או שלחתם את ילדכם?'
st_live_in_goln = '(האם אחד מבני הזוג הוא בן הגולן (גר בגולן חמש שנים ומעלה'
st_live_in_north = 'האם אתם תושבים בהווה של אחת מהמועצות הבאות: גולן, קצרין, קריית שמונה, מטולה, חצור הגלילית, ראש פינה, יסוד המעלה, גליל עליון, מבואות חרמון, מרום גליל'
st_live_in_north_res = 'זיקה גאוגרפית'
st_is_family = 'האם הינך קרוב משפחה מדרגה ראשונה של בעלי מגרש ברמת טראמפ?'
# 'אם מילאת כן בשאלה הקודמת, מי הם קרובי המשפחה?',
st_family_close_res = 'זיקה משפחתית'
st_children_num = 'כמה ילדים בגילאים 0-12 מתגוררים במשק הבית?'
st_relevant_age = 'האם יש לך ילדים באחד מהשנתונים הבאים?'
st_children_num_res = 'מספר ילדים'
st_age = 'מה גילכם?'

st_relevant_age_res = 'שנתונים מועדפים'
st_family_situation = 'תא משפחתי'
st_live_in_complex_rliges = 'האם אתם גרים או גרת/ם בעבר לפחות שנה ביישוב כפרי או בקהילת מגורים אחרת המקדמת חיי שיתוף של דתיים וחילונים כמטרה מוצהרת?'
st_family_situation_res = 'ניקוד תא משפחתי'
st_is_it_complex_cople = 'האם אתם זוג מעורב של חילוני ודתיה או להיפך?'
st_live_in_complex_rliges_res = 'גרו בקהילה מעורבת'
st_complex_parents = 'האם גדלתם בבית מעורב? (הורה חילוני והורה דתי)'
st_complex_couple_res = 'זוג מעורב'
st_complex_school = 'האם התחנכתם במוסד המוגדר על ידי משרד החינוך כמשתייך לזרם המשלב דתיים וחילונים?'
st_complex_parents_res = 'גדלו בבית מעורב'
st_complex_child_school = 'האם ילדכם רשומים או היו רשומים למוסד חינוכי המוגדר על ידי משרד החינוך כמשתייך לזרם המשלב דתיים וחילונים?'
st_complex_school_res = 'חינוך משלב הורים'
st_complex_work = 'האם עבדת או למדת במוסד ארגוני/חינוכי לבוגרים שבהגדרת מטרותיו המוצהרות שילוב דתיים וחילונים, במהלך 7 השנים האחרונות?'
st_complex_child_school_res = 'חינוך משלב ילדים'
st_complex_frame = 'האם לקחת חלק במסגרות משותפות אחרות או נוספות המשלבות דתיים וחילוניים?'
st_complex_school_or_frame = 'מוסד ארגוני או חינוכי מעורב'
# 'אם ענית כן בשאלה הקודמת יש לפרט את שמות המסגרות',
st_ather_frames = 'מסגרות משותפות אחרות'
# 'האם אתם עוסקים בהתנדבות כלשהי או לוקחים תפקיד פעיל בקהילה אליה אתם משתייכים? פרטו בקצרה',
st_complex_res = 'זיקה למעורב'
# 'אסמכתא לתשלום דמי טיפול על סך 300 ש"ח בהעברה בנקאית לאגודה קהילתית רמת טראמפ בנק לאומי, סניף 732, חשבון 10215600/08',
# 'ספחי ת.ז',
# 'אישור תושבות, או צילום של תשלום ארנונה על שימכם עם תאריך עדכני, או חוזה שכירות (למגורים בהווה באחת המועצות הצפוניות שצויינו מעלה)',
# 'לבני הגולן מכתב חתום על ידי מזכירות היישוב',
# 'למגורים בקהילה מעורבת של דתיים וחילונים מכתב חתום על ידי מזכירות הקהילה',
# 'עדות לחינוך במוסד מהזרם המשלב (הורים ו/או ילדים)- אישור רישום ממשרד החינוך, או תעודה ממוסד החינוך, או מכתב חתום ממזכירות המוסד ',
# 'עדות לעבודה או לימודים במוסד ארגוני/חינוכי לבוגרים שבהגדרת מטרותיו המוצהרות שילוב דתיים וחילונים, במהלך 7 השנים האחרונות- תלוש משכורת או מכתב חתום על ידי מזכירות המוסד, כולל התייחסות לתקופת הלימודים / התעסוקה'],


'''
אני מאשר/ת שקראתי את רשימת הקריטריונים ומצהיר/ה כי אני עומד/ת בכל תנאי הסף
שם משפחה
שם פרטי מועמד 1
מספר ת.ז. מועמד 1
מספר טלפון של מועמד 1
שם פרטי מועמד 2
מספר ת.ז. מועמד 2
מספר טלפון של מועמד 2
האם אתם שותפים באורח קבע בתפילות שחרית בשבתות בבית כנסת?
האם אתם נוסעים בשבת?
לאיזה זרם חינוכי אתם שולחים או שלחתם את ילדכם?
האם אחד מבני הזוג הינו "בן גולן" כהגדרתו במסמך זה: מקום מגוריו הקבוע במשך שש שנים ברציפות במהלך חייו היה בתחום שיפוט מועצה אזורית גולן?
האם הינך קרוב משפחה מדרגה ראשונה של בעלי מגרש ברמת טראמפ?
אם מילאת כן בשאלה הקודמת, מי הם קרובי המשפחה?
כמה ילדים בגילאים 0-12 מתגוררים במשק הבית?
האם יש לך ילדים באחד מהשנתונים הבאים?
מה גילכם?
מצב משפחתי
האם אתם גרים או גרת/ם בעבר לפחות שנה ביישוב כפרי או בקהילת מגורים אחרת המקדמת חיי שיתוף של דתיים וחילונים כמטרה מוצהרת?
האם אתם זוג מעורב של חילוני ודתיה או להיפך?
האם גדלתם בבית מעורב? (הורה חילוני והורה דתי)
האם התחנכתם במוסד המוגדר על ידי משרד החינוך כמשתייך לזרם המשלב דתיים וחילונים?
האם ילדכם רשומים או היו רשומים למוסד חינוכי המוגדר על ידי משרד החינוך כמשתייך לזרם המשלב דתיים וחילונים?
האם עבדת או למדת במוסד ארגוני/חינוכי לבוגרים שבהגדרת מטרותיו המוצהרות שילוב דתיים וחילונים, במהלך 7 השנים האחרונות?
האם לקחת חלק במסגרות משותפות אחרות או נוספות המשלבות דתיים וחילוניים?
אם ענית כן בשאלה הקודמת יש לפרט את שמות המסגרות
האם אתם עוסקים בהתנדבות כלשהי או לוקחים תפקיד פעיל בקהילה אליה אתם משתייכים? פרטו בקצרה
ספחי ת.ז
ל"בן גולן" מכתב חתום על ידי מזכירות היישוב או אישור מרישום האוכלוסין
למגורים בקהילה מעורבת של דתיים וחילונים מכתב חתום על ידי מזכירות הקהילה
עדות לחינוך במוסד מהזרם המשלב (הורים ו/או ילדים)- אישור רישום ממשרד החינוך, או תעודה ממוסד החינוך, או מכתב חתום ממזכירות המוסד



אחרי עמודה AB להוסיף עמודה עם כותרת: ניקוד מסכם זיקה לעירוב דתיים וחילונים
אחרי עמודה AA להוסיף עמודה עם כותרת: ניקוד מסגרת משלבת אחרת
אחרי עמודה Z להוסיף עמודה עם כותרת: ניקוד מוסד ארגוני / חינוכי משלב
אחרי עמודה Y להוסיף עמודה עם כותרת: ניקוד חינוך משלב ילדים
אחרי עמודה X להוסיף עמודה עם כותרת: ניקוד חינוך משלב הורים
אחרי עמודה W להוסיף עמודה עם כותרת: ניקוד גדלו בבית מעורב
אחרי עמודה V להוסיף עמודה עם כותרת: ניקוד זוג מעורב
אחרי עמודה U להוסיף עמודה עם כותרת: ניקוד יישוב מעורב

אחרי עמודה T להוסיף עמודה עם כותרת: ניקוד שנתונים חסרים
אחרי עמודה S להוסיף עמודה עם כותרת: ניקוד מספר ילדים
אחרי עמודה R להוסיף עמודה עם כותרת: ניקוד גיל מבוגר
אחרי עמודה Q להוסיף עמודה עם כותרת: ניקוד זוג צעיר ללא ילדים
אחרי עמודה P להוסיף עמודה עם כותרת: ניקוד זיקה משפחתית
אחרי עמודה M להוסיף עמודה עם כותרת: קבוצת אוכלוסיה
אחרי עמודה A להוסיף עמודה עם כותרת: ניקוד כללי
'''

st_res_complex = 'ניקוד מסכם זיקה לעירוב דתיים וחילונים'
st_res_complex_else = 'ניקוד מסגרת משלבת אחרת'
st_res_complex_school = 'ניקוד מוסד ארגוני / חינוכי משלב'

st_res_complex_school_child = 'ניקוד חינוך משלב ילדים'
st_res_complex_school_parents = ' ניקוד חינוך משלב הורים'
st_res_complex_grow_in = 'ניקוד גדלו בבית מעורב'

st_res_complex_couple = 'ניקוד זוג מעורב'
st_res_complex_village = 'ניקוד יישוב מעורב'
st_res_gup_age = 'ניקוד שנתונים חסרים'

st_res_child_num = 'ניקוד מספר ילדים'
st_res_age = 'ניקוד גיל מבוגר'
st_res_yong_no_child = 'ניקוד זוג צעיר ללא ילדים'

st_res_family_in_village = 'ניקוד זיקה משפחתית'
st_res_type = 'קבוצת אוכלוסיה'
st_res_tot = 'ניקוד כללי'


def add_columns(df):
    # df.columns.get_loc("pear")
    df.insert(28, st_res_complex, 0)
    df.insert(27, st_res_complex_else,0)
    df.insert(26, st_res_complex_school,0)
    df.insert(25, st_res_complex_school_child,0)
    df.insert(24, st_res_complex_school_parents,0)
    df.insert(23, st_res_complex_grow_in,0)
    df.insert(22, st_res_complex_couple,0)
    df.insert(21, st_res_complex_village,0)
    df.insert(20, st_res_gup_age,0)
    df.insert(19, st_res_child_num,0)
    df.insert(18, st_res_age,0)
    df.insert(17, st_res_yong_no_child,0)
    df.insert(16, st_res_family_in_village,0)
    df.insert(14, st_res_type, 0)
    df.insert(0, st_res_tot,0)

    return df


def write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, data_df):
    """
    this function takes data_df and writes it under spreadsheet_id
    and sheet_name using your credentials under service_file_path
    """
    gc = pygsheets.authorize(service_file=service_file_path)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        sh.add_worksheet(sheet_name)
    except:
        pass
    wks_write = sh.worksheet_by_title(sheet_name)
    wks_write.clear('A1',None,'*')
    wks_write.set_dataframe(data_df, (1,1), encoding='utf-8', fit=True)
    wks_write.frozen_rows = 1

def analysis(csv_input: str, res_dir: str):
    # pin_filed = 'סיסמה'
    # point_filed = 'Score'
    # good_val = '100 / 100'
    # not_relevant_column = [pin_filed, point_filed, 'Timestamp']
    if '.csv' in csv_input:
        df = pd.read_csv(csv_input, sep=",", encoding='utf-8')
    elif '.xlsx' in csv_input:
        df = pd.read_excel(csv_input)



    df = add_columns(df)

    df = df.reset_index()  # make sure indexes pair with number of rows


    for index, row in df.iterrows():
        # if row[st_live_in_goln] == st_yes or row[st_live_in_north] == st_yes:
        #     df[st_live_in_north_res] = 1

        if row[st_is_family] == st_is_parent:
            df[st_res_family_in_village][index] = 4
        elif row[st_is_family] == st_is_brother:
            df[st_res_family_in_village][index] = 1

        if row[st_children_num] == st_more_from_3:
            df[st_res_child_num][index] = 3
        else:
            df[st_res_child_num][index] = int(row[st_children_num])

        if type(row[st_relevant_age]) == str:
            df[st_res_gup_age][index] = len(str(row[st_relevant_age]).split(','))

        if row[st_age] == '50+' and row[st_is_family] != st_is_parent:
            df[st_res_age][index] = 3

        if '35' in row[st_family_situation]:
            df[st_res_yong_no_child][index] = 3


        complex_conected_points = 0
        if row[st_live_in_complex_rliges] == st_yes_both:
            complex_conected_points += 3
            df[st_res_complex_village][index] = 3
        elif row[st_live_in_complex_rliges] == st_yes_one:
            complex_conected_points += 2
            df[st_res_complex_village][index] = 2


        if row[st_complex_parents] == st_yes_both:
            complex_conected_points += 2
            df[st_res_complex_grow_in][index] = 2
        elif row[st_complex_parents] == st_yes_one:
            complex_conected_points += 1
            df[st_res_complex_grow_in][index] = 1

        if row[st_is_it_complex_cople] == st_yes:
            complex_conected_points += 2
            df[st_res_complex_couple][index] = 2


        if row[st_complex_school] == st_yes_both:
            complex_conected_points += 2
            df[st_res_complex_school_parents][index] = 2
        elif row[st_complex_school] == st_yes_one:
            complex_conected_points += 1
            df[st_res_complex_school_parents][index] = 1

        if row[st_complex_child_school] == st_yes:
            complex_conected_points += 2
            df[st_res_complex_school_child][index] = 2



        if row[st_complex_work] == st_yes_both:
            complex_conected_points += 2
            df[st_res_complex_school][index] = 2
        elif row[st_complex_work] == st_yes_one:
            complex_conected_points += 1
            df[st_res_complex_school][index] = 1

        if row[st_complex_frame] == st_yes_both:
            complex_conected_points += 2
            df[st_res_complex_else][index] = 2
        elif row[st_complex_frame] == st_yes_one:
            complex_conected_points += 1
            df[st_res_complex_else][index] = 1

        df[st_res_complex][index] = min(4, complex_conected_points)
        # a = complex_conected_points + df[st_family_situation_res][index] + df[st_res_child_num][index] + df[st_res_family_in_village][index] + df[st_live_in_north_res][index]
        # df[st_point][index] = a
    df[st_res_tot] = df[st_res_complex] + df[st_res_child_num] + df[st_res_family_in_village] + df[st_res_gup_age] + df[st_res_yong_no_child] + df[st_res_age]


    # df_dati = df[(df[st_eat_coser]==st_yes) & (df[st_prire]==st_yes)
    #     & (df[st_drive_shbat]==st_no) & (df[st_school]==st_religes_school)]
    df_dati = df[(df[st_prire]==st_yes) & (df[st_drive_shbat]==st_no) & (df[st_school]==st_religes_school)]

    df_chilony = df[(df[st_prire]==st_no) & (df[st_drive_shbat]==st_yes) & (df[st_school] == st_normal_school)]

    df_med_t = pd.merge(df, df_dati, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    df_med = pd.merge(df_med_t, df_chilony, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    # create a excel writer object
    with pd.ExcelWriter(r"C:\work_space\temp\test.xlsx") as writer:

        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df_dati.to_excel(writer, sheet_name="dati", index=False)
        df_chilony.to_excel(writer, sheet_name="chilony", index=False)
        df_med.to_excel(writer, sheet_name="med", index=False)

    # service_file_path = 'https://docs.google.com/spreadsheets/d/1RMWPKMxsd1z3Mopt39nmvap1HZFlv-Uz-HRgsaGC778/edit#gid=0'
    # spreadsheet_id = 1
    # sheet_name = 'test'
    # write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, df)




    # print('incurrect pins:', get_incorrect_pins(df, pin_filed, point_filed, good_val))
    # with open(os.path.join(res_dir, 'incorrect.txt'), 'w') as the_file:
    #     the_file.write(f'incorrect pins: {get_incorrect_pins(df, pin_filed, point_filed, good_val)}\n')
    # print('duplicate pin:', get_duplicate_pin(df, pin_filed))
    # with open(os.path.join(res_dir, 'incorrect.txt'), 'a') as the_file:
    #     the_file.write(f'duplicate pin:{get_duplicate_pin(df, pin_filed)}\n')
    #
    # df = remove_incoret_pin(df, point_filed, good_val)
    # df = fusion_same_pin(df, pin_filed)
    # # create_graph(df, not_relevant_column, res_dir)
    #
    # create_graph_plurality_block_voting(df, not_relevant_column, res_dir)
    #
    # df.to_csv(os.path.join(res_dir, 'csv_output.csv'))



if __name__ == '__main__':
    csv = r'C:\work_space\temp2\t.xlsx'
    # csv = r'C:\work_space\temp\test.csv'
    output_dir = r'C:\work_space\temp\t3'
    analysis(csv, output_dir)