unit utils;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$PointerMath on}

interface

uses
  SysUtils, lightnet, gemm, blas;

{$ifdef MSWINDOWS}
//const CLOCKS_PER_SEC: int64 =1000;
{$else}
{$endif}
const CLOCKS_PER_SEC=1000000000;

type
  TChars = set of char;
  { TTools }
  TTools<T> = record
    type
      PT = ^T;
      AT = TArray<T>;
      TComparefunc = function(const a,b:T):integer ;
    class procedure shuffle(arr: PT; const n: size_t); static;
    class procedure sorta_shuffle(const arr:PT;const n, sections: size_t);static;
    class procedure QuickSort(Arr: PT; L, R : Longint; const Compare: TComparefunc; const Descending:boolean=false);static; inline ;
  end;

function basecfg(const cfgfile:string):string;


// todo implement TIME profiling
(*

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)
*)


procedure time_random_matrix(const TA, TB, m, k, n: longint);
{$ifdef GPU}
procedure time_gpu(const TA, TB, m, k, n: integer);
{$endif}

//note print matrix?

function get_paths(const filename: string):TStringArray;

procedure print_statistics(const a:PSingle; const n:integer);

function read_intlist(const gpu_list:string; var ngpus:longint; const d:longint):TArray<longint>;//
function int_to_alphanum(const i:longint):char; //
function alphanum_to_int(const c:char):longint; //
function read_map(const filename:string):TArray<longint>;                      //
function _strip(const s:string; const chars:TChars=[' ',#9]):string;
function custom_hash(const str:string):longword;
procedure replace_image_to_label(const input_path: string; var output_path: string);

implementation

function custom_hash(const str:string):longword;
var c,i: longint;
begin
  result := 5381;
  for i:=1 to length(str) do
      result := ((result shl 5) + result) + byte(str[i]); // result * 33 + c

end;

procedure replace_image_to_label(const input_path: string; var output_path: string);
begin
    output_path := stringReplace(input_path, '/images/train2017/', '/labels/train2017/' , []);
    output_path := stringReplace(output_path, '/images/val2017/', '/labels/val2017/'    , []);
    output_path := stringReplace(output_path, '/JPEGImages/', '/labels/'                , []);
    output_path := stringReplace(output_path, '\images\train2017\', '\labels\train2017\', []);
    output_path := stringReplace(output_path, '\images\val2017\', '\labels\val2017\'    , []);
    output_path := stringReplace(output_path, '\images\train2014\', '\labels\train2014\', []);
    output_path := stringReplace(output_path, '\images\val2014\', '\labels\val2014\'    , []);
    output_path := stringReplace(output_path, '/images/train2014/', '/labels/train2014/', []);
    output_path := stringReplace(output_path, '/images/val2014/', '/labels/val2014/'    , []);
    output_path := stringReplace(output_path, '\JPEGImages\', '\labels\'                , []);
    output_path := trim(output_path);
    output_path := stringReplace(output_path, '.jpg', '.txt' ,[rfIgnoreCase]);
    output_path := stringReplace(output_path, '.jpeg', '.txt',[rfIgnoreCase]);
    output_path := stringReplace(output_path, '.png', '.txt' ,[rfIgnoreCase]);
    output_path := stringReplace(output_path, '.bmp', '.txt' ,[rfIgnoreCase]);
    output_path := stringReplace(output_path, '.ppm', '.txt' ,[rfIgnoreCase]);
    output_path := stringReplace(output_path, '.tiff', '.txt',[rfIgnoreCase]);
    if length(output_path) > 4 then
        if '.txt' <> ExtractFileExt(output_path) then
            writeln(ErrOutput, 'Failed to infer label file name (check image extension is supported): ', output_path)
    else
        writeln(ErrOutput, 'Label file name is too short: ', output_path)
end;

function read_map(const filename: string): TArray<longint>;
var
    n: longint;
    //map: TIntegers;
    str: string;
    f: TextFile;
begin
    n := 0;
    result := nil;
    if not FileExists(filename) then
      raise EFileNotFoundException.Create(filename+': not found!');
    assignFile(f,filename);
    reset(f);
    while not eof(f) do
        begin
            readln(f,str);
            inc(n);
            setLength(result, n);//.reAllocate( n );
            result[n-1] := strToInt(trim(str))
        end;
    closeFile(f);
    //exit(map)
end;

function _strip(const s:string; const chars:TChars):string;
var j:size_t;
begin
  j:=1;
  result := s;
  while j<length(result) do
    if result[j] in chars then delete(result,j,1)
    else inc(j);
end;

function get_paths(const filename: string):TStringArray;
var f:TextFile;
begin
  result:=nil;
  if not FileExists(filename) then
    raise  EFileNotFoundException.CreateFmt('File [%s] not found',[filename]);
  AssignFile(f,filename);
  reset(f);
  while not EOF(f) do begin
    setLength(result, length(result)+1);
    readln(f, result[high(result)])
  end;
  CloseFile(F);
end;

procedure print_statistics(const a: PSingle; const n: integer);
var m,v :single ;
begin
    m := mean_array(a, n);
    v := variance_array(a, n);
    writeln(format('MSE: %.6f, Mean: %.6f, Variance: %.6f', [mse_array(a, n), m, v]));
end;


function read_intlist(const gpu_list: string; var ngpus: longint;
  const d: longint): TArray<longint>;
var l:TStringArray;
    i:longint;
begin
  l := gpu_list.split([',']);
  setLength(result,length(l));
  for i:=0 to length(result)-1 do
    TryStrToInt(trim(l[i]), result[i])

end;

function int_to_alphanum(const i: longint): char;
begin

  if i = 36 then exit('.');
  if i < 10 then
    result := char(i + 48)
  else
    result := char(i + 87)
end;

function alphanum_to_int(const c: char): longint;
begin
  // todo alphanum_to_int
  if c < #58 then
    result := longint(c) - 48
  else
    result := longint(c) - 87

end;


{ TTools }

class procedure TTools<T>.shuffle( arr: PT; const n: size_t);

const RAND_MAX:longint = MaxInt;
var
    i, j: size_t;
    swp: T;
begin
    for i := 0 to n-1 do
        begin
            j := i+Random(RAND_MAX) div (RAND_MAX div (n-i)+1);
            swp    := arr[j];
            arr[j] := arr[i];
            arr[i] := swp
        end
end;

class procedure TTools<T>.sorta_shuffle(const arr: PT; const n, sections: size_t);
var i, start, &end, num:size_t;
begin
  for i := 0 to sections-1 do begin
      start := n*i div sections;
      &end := n*(i+1) div sections;
      num := &end-start;
      shuffle(@arr[start], num);
  end
end;

class procedure TTools<T>.QuickSort(Arr: PT; L, R : Longint; const Compare: TComparefunc;const Descending:boolean);
var I,J ,neg :longint;
    P, Q :T;
begin
  if not Assigned(Arr) then exit;
  
  if descending then
   neg:=-1
  else
   neg:=1;
  repeat
    I := L;
    J := R;
    P := Arr[ (L + R) shr 1 ];
    repeat
      while neg*Compare(P, Arr[i]) > 0 do
        I := I + 1;
      while neg*Compare(P, Arr[J]) < 0 do
        J := J - 1;
      If I <= J then
      begin
        Q := Arr[I];
        Arr[I] := Arr[J];
        Arr[J] := Q;
        I := I + 1;
        J := J - 1;
      end;
    until I > J;
    if J - L < R - I then
      begin
        if L < J then
          QuickSort(Arr, L, J, Compare, Descending);
        L := I;
      end
      else
      begin
        if I < R then
          QuickSort(Arr, I, R, Compare, Descending);
        R := J;
      end;
  until L >= R;
end;

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

function basecfg(const cfgfile:string):string;
var i:longint;
begin
  result := ExtractFileName(cfgfile);
  i:=pos('.',result);
  if i>0 then
    delete(result, i, length(result))
end;



// todo TA TB must be a boolean
procedure time_random_matrix(const TA, TB, m, k, n: longint);
var
  a, b, c:TSingles;
  lda, ldb, i:longint;
  start, &end : clock_t;
begin

  if not boolean(TA) then
    a := random_matrix(m,k)
  else
    a := random_matrix(k,m);
  if not boolean(TA) then lda := k else lda:= m ;
  if not boolean(TB) then
    b := random_matrix(k,n)
  else
    b := random_matrix(n,k);
  if not boolean(TB) then ldb:=n else ldb :=k;

  c := random_matrix(m,n);

  // todo replace with high res timer

  start:=clock();
  for i := 0 to 9 do
      sgemm(TA,TB,m,n,k,1,@a[0],lda,@b[0],ldb,1,@c[0],n);
  &end := clock();
  writeln(format('Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %f ms',[m,k,k,n, TA, TB, (&end-start)/CLOCKS_PER_SEC]));

end;

{$ifdef GPU}
procedure time_gpu(const TA, TB, m, k, n: integer);
var
    iter: integer;
    a: PSingle;
    b: PSingle;
    lda: integer;
    ldb: integer;
    c: PSingle;
    a_cl: PSingle;
    b_cl: PSingle;
    c_cl: PSingle;
    i: integer;
    start: clock_t;
    flop: double;
    gflop: double;
    seconds: double;
begin
    iter := 10;
    a := random_matrix(m, k);
    b := random_matrix(k, n);
    if (not boolean(TA)) then
        lda := k
    else
        lda := m;
    if (not boolean(TB)) then
        ldb := n
    else
        ldb := k;
    c := random_matrix(m, n);
    a_cl := cuda_make_array(a, m * k);
    b_cl := cuda_make_array(b, k * n);
    c_cl := cuda_make_array(c, m * n);
    start := clock();
    for i := 0 to iter -1 do
        begin
            gemm_gpu(TA, TB, m, n, k, 1, a_cl, lda, b_cl, ldb, 1, c_cl, n);
            cudaThreadSynchronize()
        end;
    flop := m * n * (2. * k+2.) * iter;
    gflop := flop / pow(10., 9);
    &end := clock();
    seconds := sec(&end-start);
    writeln(format('Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %f s, %f GFLOPS', [m, k, k, n, TA, TB, seconds, gflop / seconds]));
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
end;
{$endif}



end.

