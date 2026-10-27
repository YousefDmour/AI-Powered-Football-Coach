We reviewed the match event dataset and confirmed the Barcelona pass schema, then built and verified a pass-accuracy analysis showing 917 passes with 820 successful and an overall accuracy of 89.42%. The project now includes a working analysis script and a regression test to ensure the Barcelona pass metrics remain accurate as the project evolves. 

Added player and team coaching recommendations based on passing, shooting, dribbling, duels, and defensive metrics.
Added a simple console report in visualization/player_report.py showing each player's name, metrics, and recommendations.
Validated the recommendation system and report with the bundled match data; all recommendation tests passed.
## 2026-10-27

- أضفنا وحدة `vision/detect_players.py` لاستخراج إطار من الفيديو باستخدام OpenCV.
- أضفنا كشف اللاعبين باستخدام نموذج YOLOv8 مع فلترة الأشخاص، وإظهار درجة الثقة ومربعات الكشف.
- أضفنا رسم نتائج الكشف وحفظها في `vision/output.jpg`، مع توفير فيديو تجريبي `vision/sample.mp4` وملف النموذج `yolov8n.pt`.
- حدّثنا تقرير اللاعبين في `visualization/player_report.py` لعرض تسمية اللاعب باللغة الإنجليزية ضمن التقرير.
- أضفنا `vision/track_players.py` لتتبّع اللاعبين عبر الفيديو باستخدام YOLO وBoT-SORT، مع دعم ByteTrack وإعدادات مثل `track_buffer` و`conf` و`imgsz`.
- أضفنا حفظ بيانات التتبّع بصيغة CSV، وتلخيص عدد المعرفات وطول المسارات ومتوسط الاكتشافات لكل إطار، وتسجيل نتائج التجارب في `vision/experiments.csv`.
- شغّلنا تجربة التتبّع الأساسية على 250 إطارًا، وسجّلنا 106 معرفات للاعبين، منها 45 مسارًا بطول 25 إطارًا على الأقل و15 مسارًا بطول 100 إطار على الأقل.
- أنشأنا فيديو التتبّع `vision/tracked_baseline.mp4` وملف النتائج `vision/tracks_baseline.csv`، وأضفنا نموذج YOLOv8 متوسط الحجم `yolov8m.pt` للتجارب القادمة.
