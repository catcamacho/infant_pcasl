#! /bin/csh

set asl_dir = /Users/catcamacho/Box/SNAP/BABIES/Raw
set raw_dir = /Users/catcamacho/Box/SNAP/BABIES/subjDir

foreach sub (002 002x 010 012 020 021 023 025 027 028x 031 032 033x 035 036 039 040 045 056 061 062 064x 067 072 076x 077x 087)

set asl = ${sub}-C-T1
set dest = ${sub}-BABIES-T1

if (-e $asl_dir/$asl/pd.nii) then
	if (-e $raw_dir/$dest) then
		mv $asl_dir/$asl/pw.nii $raw_dir/$dest/pw.nii
		mv $asl_dir/$asl/pd.nii $raw_dir/$dest/pd.nii
	else
		mkdir $raw_dir/$dest
		mv $asl_dir/$asl/pw.nii $raw_dir/$dest/pw.nii
		mv $asl_dir/$asl/pd.nii $raw_dir/$dest/pd.nii
		mv $asl_dir/$asl/processed_t1.nii $raw_dir/$dest/skullstripped_anat.nii
	endif
endif
end