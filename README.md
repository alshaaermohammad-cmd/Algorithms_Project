# Algorithms_Project

## اختيار المشاريع ضمن ميزانية محددة

**BIA601 — Intelligent Algorithms / الخوارزميات الذكية — الوظيفة الأولى (F25) — المجموعة 2**

المستودع: https://github.com/alshaaermohammad-cmd/Algorithms_Project

مشروع اختيار مشاريع ضمن ميزانية لا تكفي لتنفيذها كلها (مسألة الحقيبة الثنائية 0/1). نقارن خوارزمية جينية (GA) وأسراب الجسيمات الثنائية (PSO) ومستعمرة النمل (ACO) مع حل دقيق بالبرمجة الديناميكية (DP) وحل جشع.

0/1 knapsack project selection under a budget constraint, comparing GA, binary PSO, and ACO with an exact DP baseline and a greedy heuristic. All three intelligent algorithms are implemented from scratch in this repository.

## فريق العمل — Group 2

| العضو | GitHub | المسؤولية | القضية |
|---|---|---|---|
| عبدالسلام أبو جمعة | [Abed99](https://github.com/Abed99) | تعريف المسألة والبيانات والتصليح | [#1](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/1) |
| عبدالعزيز حسام موصللي | [azzooz12321](https://github.com/azzooz12321) | خوارزمية GA | [#2](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/2) |
| محمد الشراري | [mohamedalshrari2-cyber](https://github.com/mohamedalshrari2-cyber) | خوارزمية PSO | [#3](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/3) |
| آية حسن | [ayahasan001](https://github.com/ayahasan001) | خوارزمية ACO | [#4](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/4) |
| محمد هادي الغبرة | [alghabrahadi12-bit](https://github.com/alghabrahadi12-bit) | DP و Greedy والتحقق من القيود | [#5](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/5) |
| عمر محمود الهمشري | [Omar48966](https://github.com/Omar48966) | ضبط البارامترات والتجارب | [#6](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/6) |
| جواد اجرزو | [jwadjrzw-netizen](https://github.com/jwadjrzw-netizen) | الرسوم البيانية وتصدير النتائج | [#7](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/7) |
| اية محمد العلوش | [alooshaya05-boop](https://github.com/alooshaya05-boop) | التقرير ودليل التشغيل | [#8](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/8) |
| محمد خالد الشاعر | [alshaaermohammad-cmd](https://github.com/alshaaermohammad-cmd) | التكامل النهائي والتوثيق وواجهة العرض | [#9](https://github.com/alshaaermohammad-cmd/Algorithms_Project/issues/9) |

مساهمة كل عضو موثّقة في التقرير (`Report.docx`، جدول 10) وفي تاريخ المستودع.

## التشغيل

Python 3.12 أو أحدث، من مجلد `project`:

```text
python -m pip install -r requirements.txt
python main.py
python verify.py
```

النتائج المرفقة تخص 20 تشغيلة لكل خوارزمية ولكل مسألة. تشغيل الضبط والتجارب الكاملة يستغرق عدة دقائق.

تجربة قصيرة تكتب في مجلد مستقل، دون تعديل النتائج المرفقة:

```text
python main.py --runs 2 --output ../trial_results
```

إعادة أول تشغيلة (الثابت 0) لكل خوارزمية ولكل مسألة، مع مقارنة الحلول ومسارات التقارب بالنتائج المرفقة:

```text
python verify.py --rerun
```

التفاصيل التقنية والإعدادات والتجارب موجودة في [`project/README.md`](project/README.md).

## النتائج والعرض

- التقرير: [`Report.docx`](Report.docx) ونسخة PDF للتسليم: [`Report.pdf`](Report.pdf)
- لوحة النتائج التفاعلية: افتح [`index.html`](index.html) في المتصفح (ملف مستقل، لا يحتاج تشغيل خادم).
  لعرض الجداول والرسوم بعد التجربة الكاملة يكفي فتح الملف محلياً.
- مخرجات التجارب والرسوم: [`project/results/`](project/results/)

## مخرجات التسليم

المشروع جاهز للتسليم قبل 12 نوفمبر 2026، ويتضمن:

1. الكود المصدري القابل للتشغيل في `project/` (GA و PSO و ACO من الصفر، مع DP وحل جشع للمقارنة).
2. تقريراً مختصراً يشرح المسألة والخوارزميات والتجارب والنتائج ومهمة كل عضو (`Report.docx` و`Report.pdf`).
3. ملف `README.md` هذا مع تعليمات التشغيل وإعادة التجربة عبر `verify.py`.
