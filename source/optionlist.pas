unit OptionList;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface

uses
  SysUtils, darknet, data, list;

type
  PKvp = ^TKvp;
  TKvp = record
    key : string;
    val : string;
    used : boolean
  end;

  PSection = ^TSection;
  TSection = record
    &type : string;
    options :TList
  end;


function read_data_cfg(const filename: string):TList;//PList;
function get_metadata(const filename: string):TMetadata;
function parseKV(const s: string; var options: TList):boolean;
procedure option_insert(var l: TList; key, val: string);
procedure option_unused(const l: TList);
function option_find(const l: TList; const key: string):string;
function option_find_str(const l: TList; const key, def: string):string;
function option_find_str_quiet(const l: TList; const key, def: string):string;
function option_find_int(const l: TList; const key: string; const def: longint):longint;            overload;
function option_find_int_quiet(const l: TList; const key: string; const def: longint):longint;      overload;
function option_find_bool(const l: TList; const key: string; const def: boolean):boolean;            overload;
function option_find_bool_quiet(const l: TList; const key: string; const def: boolean):boolean;      overload;

function option_find_float_quiet(const l: TList; const key: string; const def: Single):single;
function option_find_float(const l: TList; const key: string; const def: single):single;
procedure free_section(var s: PSection);

implementation

procedure free_section(var s: PSection);
var
    n, next: PNode;
    pair : PKvp;
begin
    //free(s.&type);
    n := s.options.front;
    while assigned(n) do
        begin
            pair := PKvp(n.val);
            pair.key:='';
            pair.val:='';
            //free(pair.key);
            FreeMemAndNil(pair);
            next := n.next;
            FreeMemAndNil(n);
            n := next
        end;
    s.&type:='';
    //FreeMemAndNil(s.options);
    FreeMemAndNil(s)
end;

function read_data_cfg(const filename: string): TList;
var
    f: TextFile;
    line: string;
    nu: longint;
    options: TList;
begin
    if not FileExists(filename) then
        raise EFileNotFoundException.Create(filename+': was not found');
    result:=default(TList);
    AssignFile(f , filename);
    reset(f);
    nu := 0;
    options := Default(TList);//make_list();
    while not EOF(f) do
        begin
            readln(f,line);
            inc(nu);
            line:=trim(line);
            if (line='') or (line[1] in ['#', ';']) then
                  continue
            else
                if not parseKV(line, options) then
                    begin
                        writeln(ErrOutput, format('Config f error line %d, could parse: %s', [nu, line]));
                        //free(line)
                    end
        end;
    closeFile(f);
    exit(options)
end;

function get_metadata(const filename: string):TMetadata;
var
    //m: TMetadata;
    options: TList;
    name_list: string;
begin
    result := default(TMetadata);
    options := read_data_cfg(filename);
    name_list := option_find_str(options, 'names', '');
    if name_list='' then
        name_list := option_find_str(options, 'labels', '');
    if name_list='' then
        writeln(ErrOutput, 'No names or labels found')
    else
        result.names := get_labels(name_list);
    result.classes := option_find_int(options, 'classes', 2);
    //free_list(options);
    if(name_list<>'') then begin
        writeln(format('Loaded - names_list: %s, classes = %d ', [name_list, result.classes]));
    end;

    //exit(result)
end;

function parseKV(const s: string; var options: TList): boolean;
const eq ='=';
var
    i,j: size_t;
    val, key: string;
begin
    result := false;
    i:=pos(eq, s );
    if i=0 then exit;
    //key:=_strip(copy(s,1,i-1));

    //val:=_strip(copy(s,i+length(eq)));
    key:=trim(copy(s,1,i-1));
    val:=trim(copy(s,i+length(eq)));
    option_insert(options, key, val);

    result:=true
end;

procedure option_insert(var l: TList; key, val: string);
var
    p: PKvp;
begin
    p := AllocMem(sizeof(TKvp));
    p.key := key;
    p.val := val;
    p.used := false;
    list_insert(l, p)
end;

procedure option_unused(const l: TList);
var
    n: PNode;
    p : PKvp;
begin
    n := l.front;
    while assigned(n) do
        begin
            p := PKvp(n.val);
            if not p.used then
                writeln(ErrOutput, format('Unused field: ''%s = %s''', [p.key, p.val]));
            n := n.next
        end
end;

function option_find(const l: TList; const key: string): string;
var
    n:PNode;
    p : PKvp;
begin
    result :='';
    n := l.front;
    while assigned(n) do
        begin
            p := Pkvp(n.val);
            if p.key = key then
                begin
                    p.used := true;
                    exit(p.val)
                end;
            n := n.next
        end;

end;

function option_find_str_quiet(const l: TList; const key, def: string): string;
begin
    result := option_find(l, key);
    if result<>'' then
        exit(result);
    result:=def;
end;

function option_find_str(const l: TList; const key, def: string): string;
begin
    result := option_find(l, key);
    if result<>'' then
        exit(result);
    if def<>'' then
        writeln(ErrOutput, format('%s: Using default ''%s'''#10'', [key, def]));
    result:=def;
end;

function option_find_int(const l: TList; const key: string; const def: longint
  ): longint;
var
    v: string;
begin
    v := option_find(l, key);
    if v<>'' then
        exit(strToInt(v));
    writeln(ErrOutput, format('%s: Using default ''%d''', [key, def]));
    result:=def
end;

function option_find_int_quiet(const l: TList; const key: string;
  const def: longint): longint;
var
    v: string;
begin
    v := option_find(l, key);
    if v<>'' then
        exit(strToInt(v));
    result := def
end;

function option_find_bool(const l: TList; const key: string; const def: boolean
  ): boolean;
begin
  result := boolean(option_find_int(l,key,longint(def)))
end;

function option_find_bool_quiet(const l: TList; const key: string;
  const def: boolean): boolean;
begin
    result := boolean(option_find_int_quiet(l,key,longint(def)))
end;

function option_find_float_quiet(const l: TList; const key: string;
  const def: Single): single;
var
    v: string;
begin
    v := option_find(l, key);
    if v<>'' then
        exit(StrToFloat(v));
    result := def
end;

function option_find_float(const l: TList; const key: string; const def: single
  ): single;
var
    v: string;
begin
    v := option_find(l, key);
    if v<>'' then
        exit(StrToFloat(v));
    writeln(ErrOutput, format('%s: Using default ''%f''', [key, def]));
    result := def
end;


end.

