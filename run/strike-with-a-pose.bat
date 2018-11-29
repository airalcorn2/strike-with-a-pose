0<0# : ^
'''
@echo off
echo batch code
python %~f0 %*
exit /b 0
'''

from strike_with_a_pose import app

app.run_gui()