@ echo off
echo
echo LPR Batch Processor (by: dan.clementi@idt.com)
echo Modified by Aras Pirbadian (Aras.Pirbadian@gmail.com)
echo ....processing NEW .txt files only
echo ......(hint: if you want to force re-processing, just delete the previous .csv file)
for %%f in (*.txt) do if not exist %%f.csv call pythonLPR_2014.bat "%%f" 
:end
