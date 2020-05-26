#!/bin/bash

#------------------------------------------------------------------------------
declare -a missing_pypkg
missing_pypkg=()

function chkpypkg() {
  if python -c "import pkgutil; raise SystemExit(1 if pkgutil.find_loader('${1}') is None else 0)" &> /dev/null; then

    if [[ ! -z "$2" ]]; then
        if python -c "import ${1}; assert ${1}.__version__=='${2}'" &> /dev/null; then
            echo "${1} (${2}) is installed"
        else
            echo "${1} - wrong version installed - expected ${2}, detected $(python -c "import ${1}; print ${1}.__version__")"
            missing_pypkg+=(${1})
        fi
    else
        echo "${1} is installed"
    fi

else
    echo "Error: package '${1}' is not installed"
    missing_pypkg+=(${1})
fi
}
#------------------------------------------------------------------------------

chkpypkg xgboost
chkpypkg sklearn
chkpypkg numpy
chkpypkg pandas
chkpypkg pickle
chkpypkg tkinter
chkpypkg yfinance
chkpypkg quandl
chkpypkg math
chkpypkg csv
chkpypkg shutil
chkpypkg copy
chkpypkg time
chkpypkg datetime
chkpypkg requests
chkpypkg warnings
chkpypkg logging
chkpypkg joblib
chkpypkg scipy
chkpypkg h5py
chkpypkg future
chkpypkg matplotlib
chkpypkg seaborn
chkpypkg tensorflow
chkpypkg keras

(( ${#missing_pypkg[@]} > 0 )) &&  return 1
unset missing_pypkg

# get installation location
# SCRIPT=$("realpath -s $BASH_SOURCE")
INSTALL_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export DF_ROOT="$INSTALL_LOC"

echo "Installation location :" $INSTALL_LOC



