clc;
clear;
close all;
tic
dbstop if error
Path = 'D:\程序';
data = importdata([Path filesep '312aalmeanfeaturedata.xlsx']);
[subNum,featureNum] = size(data);
data=xlsread("312aalmeanfeaturedata.xlsx");

a2=xlsread("884data.xlsx");

site = a2(:,2);
TR = a2(:,5);
sex = a2(:,7);
age  = a2(:,6);
Ddata = combat(data(2:end,:).',sex,age,1);
Ddata = combat(Ddata,TR,age,1);
Ddata = combat(Ddata,site,age,1);
Ddata = Ddata';
dlmwrite([Path filesep 'intensity_combat_eyes_TR_site_new312.txt'], Ddata, 'delimiter' , ' ' , 'precision', '%0.4f');