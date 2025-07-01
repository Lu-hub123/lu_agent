
#region conda initialize
# !! Contents within this block are managed by 'conda init' !!
If (Test-Path "D:\Anaconda\anaconda\Scripts\conda.exe") {
    (& "D:\Anaconda\anaconda\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | ?{$_} | Invoke-Expression
}
#endregion

