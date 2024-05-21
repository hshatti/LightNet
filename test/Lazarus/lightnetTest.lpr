program darknetTest;

{$mode objfpc}{$H+}
{.$ifdef DEBUG}
{$apptype console}
{.$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  {$IFDEF HASAMIGA}
  athreads,
  {$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, SysUtils, uMain, uimgform, ntensors;

{$R *.res}
const heaptrcFile ='heap.trc';
begin
  {$if Declared(UseHeapTrace)}
  if FileExists(heaptrcFile) then DeleteFile(heaptrcFile);
  SetHeapTraceOutput(heaptrcFile);
  {$ifend}
  RequireDerivedFormResource:=True;
  Application.Scaled:=True;
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TImgForm, ImgForm);
  Application.Run;
end.

