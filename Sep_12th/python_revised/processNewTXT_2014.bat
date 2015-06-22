@ echo off
echo
echo LPR Batch Processor (by: dan.clementi@idt.com)
echo ....processing NEW .txt files only
echo ......(hint: if you want to force re-processing, just delete the previous .csv file)
for %%f in (*.txt) do if not exist %%f.csv call C:\Users\apirbadi\Documents\Meetings\Sep_12th\python_revised\pythonLPR_2014.bat "%%f" 
:end
