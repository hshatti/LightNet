unit list;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface

uses
  SysUtils, darknet;

type
  PList = ^TList;
  TList = record
      size : longint;
      front : PNode;
      back : PNode;
    end;


function make_list():PList;
function list_pop(var l: TList):Pointer;
procedure list_insert(var l: TList; const val: Pointer);
procedure free_node(n: PNode);
procedure free_list(l: PList);
procedure free_list_contents(const l: TList);
procedure free_list_contents_kvp(var l:TList);
function list_to_array(const l: PList):TArray<Pointer>;

implementation
uses OptionList;
function make_list():PList;
begin
    result := AllocMem(sizeof(TList)); //todo [make_list] AllocMem is enough though
    result.size := 0;
    result.front := nil;
    result.back := nil;
end;

function list_pop(var l: TList): Pointer;
var
    b: PNode;
begin
    if not assigned(l.back) then
        exit(nil);
    b := l.back;
    result := b.val;
    l.back := b.prev;
    if assigned(l.back) then
        l.back.next := nil;
    Freemem(b);
    dec(l.size);
end;

procedure list_insert(var l: TList; const val: Pointer);
var
    _new: PNode;
begin
    _new := AllocMem(sizeof(TNode));
    _new.val := val;
    _new.next := nil;
    if not Assigned(l.back) then
        begin
            l.front := _new;
            _new.prev := nil
        end
    else
        begin
            l.back.next := _new;
            _new.prev := l.back
        end;
    l.back := _new;
    inc(l.size)
end;

procedure free_node(n: PNode);
var
    next: PNode;
begin
    while (assigned(n)) do
        begin
            next := n.next;
            FreeMemAndNil(n);
            n := next
        end
end;

procedure free_list(l: PList);
begin
    free_node(l.front);
    FreeMemAndNil(l)
end;

procedure free_list_contents(const l: TList);
var
    n: PNode;
begin
    n := l.front;
    while (assigned(n)) do
        begin
            FreeMemAndNil(n.val);
            n := n.next
        end
end;

procedure free_list_contents_kvp(var l: TList);
var n :PNode;
    p :PKvp;
begin
    n := l.front;
    while assigned(n) do begin
        p := PKvp(n.val);
        FreeMemAndNil(p.key);
        FreeMemAndNil(n.val);
        n := n.next;
    end;

end;

function list_to_array(const l: PList):TArray<Pointer>;
var
    count: longint;
    n: PNode;
begin
    setLength(result,l.size);
    count := 0;
    n := l.front;
    while (assigned(n)) do
        begin
            result[count] := n.val;
            n := n.next;
            inc(count)
        end;
end;


end.

