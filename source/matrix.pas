unit matrix;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}
{$writeableconst on}

interface

uses
  SysUtils, lightnet, blas;

procedure pm(M: longint; N: longint; A: PSingle);
procedure free_matrix(const m: TMatrix);
function matrix_topk_accuracy(const truth, guess: TMatrix; const k: longint):single;
procedure scale_matrix(const m: TMatrix; const scale: single);
function resize_matrix(var m: TMatrix; const size: longint):TMatrix;
procedure matrix_add_matrix(const from, &to: TMatrix);
function copy_matrix(const m: TMatrix):TMatrix;
function make_matrix(const rows, cols: longint):TMatrix;
function hold_out_matrix(var m: TMatrix; const n: longint):TMatrix;
function pop_column(var m: TMatrix; const c: longint):TArray<single>;//TSingles;
function csv_to_matrix(const filename: string):TMatrix;
procedure matrix_to_csv(const m: TMatrix);
procedure print_matrix(const m: TMatrix);



implementation

procedure pm(M: longint; N: longint; A: PSingle);
var i, j:longint;
begin
  for i := 0 to M-1 do begin
      write(i+1,' ');
      for j := 0 to N-1 do
          write(A[i*N+j]:2:4, ' ');
      writeln('');
  end;
  writeln('')
end;

function count_fields(const line:string):longint;
var i:longint;
begin
  result:=0;
  if trim(line)='' then exit;
  for i:=1 to length(line) do
    if line[i]=',' then
      inc(result);
  inc(result)
end;

procedure parse_fields(const line: string; const n: longint;var result:TArray<single>); overload;
const
   l: longint = 1;
   p: longint = 0;
var
   i:integer;
   s:string;
begin
  if trim(line)='' then exit();
  //result:=TSingles.Create(n);
  setLength(result, n);
  for i:=1 to length(line) do
    if line[i]=',' then
      begin
        s:=trim(copy(line,l,i-l));
        if s<>'' then
          result[p]:=StrToFloat(s);
        l:=i+1;
        inc(p)
      end;
  s:=trim(copy(s,l));
  if s<>'' then
    result[p]:=StrToFloat(s);

  // todo test parse_fields
end;

function parse_fields(const line: string; const n: longint): TSingles;               overload;
const
   l: longint = 1;
   p: longint = 0;
var
   i:integer;
   s:string;
begin
  if trim(line)='' then exit(nil);
  result:=TSingles.Create(n);
  for i:=1 to length(line) do
    if line[i]=',' then
      begin
        s:=trim(copy(line,l,i-l));
        if s<>'' then
          result[p]:=StrToFloat(s);
        l:=i+1;
        inc(p)
      end;
  s:=trim(copy(s,l));
  if s<>'' then
    result[p]:=StrToFloat(s);

  // todo test parse_fields
end;

procedure free_matrix(const m: TMatrix);
var
    i: longint;
begin
    //for i := 0 to m.rows -1 do
        //m.vals[i].free;
    //freemem(m.vals)
end;

function matrix_topk_accuracy(const truth, guess: TMatrix; const k: longint):single;
var
    indexes: TArray<longint>;
    n, i, j, correct, &class: longint;
begin
    setLength(indexes, k);
    n := truth.cols;
    correct := 0;
    for i := 0 to truth.rows -1 do
        begin
            top_k(@guess.vals[i][0], n, k, @indexes[0]);
            for j := 0 to k -1 do
                begin
                    &class := indexes[j];
                    if truth.vals[i][&class]<>0 then
                        begin
                            inc(correct);
                            break
                        end
                end
        end;
    //free(indexes);
    exit(correct / truth.rows)
end;

procedure scale_matrix(const m: TMatrix; const scale: single);
var
    i, j: longint;
begin
    for i := 0 to m.rows -1 do
        for j := 0 to m.cols -1 do
            m.vals[i][j] := m.vals[i][j] * scale
end;

function resize_matrix(var m: TMatrix; const size: longint):TMatrix;
var
    i: longint;
begin
    if (m.rows = size) then
        exit(m);
    if (m.rows < size) then
        begin
            setLength(m.vals, size);
            //m.vals := ReAllocMem(m.vals, size * sizeof(TSingles));
            for i := m.rows to size -1 do
                setLength(m.vals[i], m.cols);// := TSingles.Create(m.cols)
        end
    else
        if m.rows > size then
            begin
                //for i := size to m.rows -1 do
                    //m.vals[i].free;
              setLength(m.vals, size);// := ReAllocMem(m.vals, size * sizeof(TSingles))
            end;
    m.rows := size;
    exit(m)
end;

procedure matrix_add_matrix(const from, &to: TMatrix);
var
    i, j: longint;
begin
    assert((from.rows = &to.rows) and (from.cols = &to.cols));
    for i := 0 to from.rows -1 do
        for j := 0 to from.cols -1 do
            &to.vals[i][j] := &to.vals[i][j] + from.vals[i][j]
end;

function copy_matrix(const m: TMatrix):TMatrix;
var
    i: longint;
begin
    result := default(TMatrix);
    result.rows := m.rows;
    result.cols := m.cols;
    setLength(result.vals, result.rows ); //:= AllocMem(result.rows * sizeof(TSingles));
    for i := 0 to result.rows -1 do
        begin
            setLength(result.vals[i], result.cols);// := TSingles.Create(result.cols);
            copy_cpu(result.cols, @m.vals[i][0], 1, @result.vals[i][0], 1)
        end;
    //exit(result)
end;

function make_matrix(const rows, cols: longint):TMatrix;
var
    i: longint;
begin
    result.rows := rows;
    result.cols := cols;
    setLength(result.vals, result.rows);// := AllocMem(result.rows* sizeof(TSingles));
    for i := 0 to result.rows -1 do
        setLength(result.vals[i],result.cols);// := TSingles.Create(result.cols);
    //exit(result)
end;

function hold_out_matrix(var m: TMatrix; const n: longint):TMatrix;
var
    i, index: longint;
begin
    result.rows := n;
    result.cols := m.cols;
    setLength(result.vals, result.rows);// := AllocMem(result.rows* sizeof(TSingles));
    for i := 0 to n -1 do
        begin
            index := random(m.rows);
            result.vals[i] := m.vals[index];
            // todo [hold_out_matrix] possible memory leak
            m.vals[index] := m.vals[_ced(m.rows)]
        end;
    //exit(result)
end;

function pop_column(var m: TMatrix; const c: longint): TArray<single>;
var
    i, j: longint;
begin
    setLength(result, m.rows);// := TSingles.Create(m.rows);
    for i := 0 to m.rows -1 do
        begin
            result[i] := m.vals[i][c];
            for j := c to m.cols-1 -1 do
                m.vals[i][j] := m.vals[i][j+1]
        end;
    dec(m.cols);
    //exit(result)
end;

function csv_to_matrix(const filename: string):TMatrix;
var
    n, size: longint;
    line:string ;
    fp: TextFile;
begin
    if not FileExists(Filename)then
        raise EFileNotFoundException.Create(filename+': file not found!');
    AssignFile(fp, filename);
    reset(fp);
    result.cols := -1;
    n := 0;
    size := 1024;
    setLength(result.vals , size);
    //result.vals := AllocMem(size* sizeof(TSingles));
    while not EOF(Fp) do
        begin
            readln(fp,line);
            if result.cols = -1 then
                result.cols := count_fields(line);
            if n = size then
                begin
                    size := size * 2;
                    setLength(result.vals, size);
                    //result.vals := ReAllocMem(result.vals, size * sizeof(TSingles))
                end;
            //result.vals[n] := parse_fields(line, result.cols);
            parse_fields(line,result.cols, result.vals[n]);
            //free(line);
            inc(n)
        end;
    setLength(result.vals , n);
    //result.vals := ReAllocMem(result.vals, n * sizeof(TSingles));
    result.rows := n;
    CloseFile(fp)
end;

procedure matrix_to_csv(const m: TMatrix);
var
    i, j: longint;
begin
    for i := 0 to m.rows -1 do
        begin
            for j := 0 to m.cols -1 do
                begin
                    if j > 0 then
                        write(',');
                    write(format('%.17g', [m.vals[i][j]]))
                end;
            writeln('')
        end
end;

procedure print_matrix(const m: TMatrix);
var
    i, j: longint;
begin
    writeln('%d X %d Matrix:', m.rows, m.cols);
    write(' __');
    for j := 0 to 16 * m.cols-1 -1 do
        write(' ');
    writeln('__ ');
    write('|  ');
    for j := 0 to 16 * m.cols-1 -1 do
        write(' ');
    writeln('  |');
    for i := 0 to m.rows -1 do
        begin
            write('|  ');
            for j := 0 to m.cols -1 do
                write(format('%15.7f ', [m.vals[i][j]]));
            writeln(' |')
        end;
    write('|__');
    for j := 0 to 16 * m.cols-1 -1 do
        write(' ');
    writeln('__|')
end;


end.

