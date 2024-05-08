unit uimgform;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, Buttons, lightnet, imagedata, Types;

type

  { TImgForm }

  TImgForm = class(TForm)
    BitBtn1: TBitBtn;
    BitBtn2: TBitBtn;
    btnCenter: TBitBtn;
    ImageList1: TImageList;
    img: TImage;
    Panel1: TPanel;
    SaveDialog1: TSaveDialog;
    ScrollBox1: TScrollBox;
    procedure BitBtn1Click(Sender: TObject);
    procedure BitBtn2Click(Sender: TObject);
    procedure btnCenterClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure imgMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    class function ShowImage(const im: TImageData; const Modal: boolean): TImgForm; static;
  private

  public

  end;

var
  ImgForm: TImgForm;

implementation

{$R *.lfm}

{ TImgForm }

procedure TImgForm.btnCenterClick(Sender: TObject);
begin
  if (img.Picture.Graphic<>nil) and SaveDialog1.Execute then
    img.Picture.SaveToFile(SaveDialog1.FileName);
end;

procedure TImgForm.BitBtn1Click(Sender: TObject);
begin
  ScrollBox1.ScaleBy(11,10);
end;

procedure TImgForm.BitBtn2Click(Sender: TObject);
begin
  ScrollBox1.scaleby(-11,10);
end;

procedure TImgForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  closeAction := caFree;
end;

procedure TImgForm.imgMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
begin
  //if WheelDelta>0 then
  //  ScrollBox1.ScaleBy(-11,10)
  //else
  //  ScrollBox1.ScaleBy(11,10);
end;

class function TImgForm.ShowImage(const im: TImageData; const Modal: boolean
  ): TImgForm;
var bmp:TBitmap;
begin
  result:=TImgForm.Create(nil);
  bmp:=imageToBitmap(im);
  result.img.Width := bmp.Width;
  result.img.Height := bmp.Height;
  result.Position:=poScreenCenter;
  result.img.Picture.Graphic:=bmp;
  FreeAndNil(bmp);
  result.img.Center:=true;
  result.img.AutoSize:=true;
  Application.ProcessMessages;
  result.img.Stretch:=true;
  if Modal then
    result.ShowModal
  else
    result.Show;
end;

end.

