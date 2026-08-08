; Inno Setup script for OpenFlow.
; Builds a single OpenFlowSetup.exe that installs the PyInstaller output
; from dist\OpenFlow and creates Start Menu / Desktop shortcuts.

#define MyAppName "OpenFlow"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "OpenFlow"
#define MyAppExeName "OpenFlow.exe"

[Setup]
AppId={{B6C1E6C0-6E9B-4C7A-9C8A-6C1E6C0B6C1E}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist_installer
OutputBaseFilename=OpenFlowSetup
Compression=lzma2
SolidCompression=yes
SetupIconFile=assets\openflow.ico
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\OpenFlow\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
