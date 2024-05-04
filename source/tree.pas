unit tree;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, lightnet, utils;

procedure change_leaves(const t: PTree; const leaf_list: string);
function get_hierarchy_probability(const x: TSingles; const hier: TTree; c: longint; const stride: longint=1):single;
procedure hierarchy_predictions(const predictions: TSingles; const n: longint; const hier: TTree; const only_leaves:boolean ; const stride: longint=1);
function hierarchy_top_prediction(const predictions: TSingles; const hier: TTree; const thresh: single; const stride: longint):longint;
function read_tree(const filename: string):TTree;

implementation

procedure change_leaves(const t: PTree; const leaf_list: string);
var
    //llist: PLise;
    leaves: TArray<string>;
    n ,i ,j ,found: longint;
begin
    //llist := get_paths(leaf_list);
    leaves := get_paths(leaf_list);//TArray<string>(list_to_array(llist));
    n := length(leaves);//llist.size;
    found := 0;
    for i := 0 to t.n -1 do
        begin
            t.leaf[i] := 0;
            for j := 0 to n -1 do
                if t.name[i]= leaves[j] then
                    begin
                        t.leaf[i] := 1;
                        inc(found);
                        break
                    end
        end;
    writeln(ErrOutput, format('Found %d leaves.', [found]))
end;

function get_hierarchy_probability(const x: TSingles; const hier: TTree;
  c: longint; const stride: longint): single;
begin
    result := 1;
    while (c >= 0) do
        begin
            result := result * x[c * stride];
            c := hier.parent[c]
        end;
    //exit(result)
end;

procedure hierarchy_predictions(const predictions: TSingles; const n: longint;
  const hier: TTree; const only_leaves: boolean; const stride: longint);
var
    j: longint;
    parent: longint;
begin
    for j := 0 to n -1 do
        begin
            parent := hier.parent[j];
            if parent >= 0 then
                predictions[j * stride] := predictions[j * stride] * predictions[parent * stride]
        end;
    if only_leaves then
        for j := 0 to n -1 do
            if hier.leaf[j]=0 then
                predictions[j * stride] := 0
end;

function hierarchy_top_prediction(const predictions: TSingles;
  const hier: TTree; const thresh: single; const stride: longint): longint;
var
    group, i, max_i, index: longint;
    p, _max, val: single;
begin
    p := 1;
    group := 0;
    while true do
        begin
            _max := 0;
            max_i := 0;
            for i := 0 to hier.group_size[group] -1 do
                begin
                    index := i+hier.group_offset[group];
                    val := predictions[(i+hier.group_offset[group]) * stride];
                    if val > _max then
                        begin
                            max_i := index;
                            _max := val
                        end
                end;
            if p * _max > thresh then
                begin
                    p := p * _max;
                    group := hier.child[max_i];
                    if hier.child[max_i] < 0 then
                        exit(max_i)
                end
            else
                if group = 0 then
                    exit(max_i)
            else
                exit(hier.parent[hier.group_offset[group]])
        end;
    result := 0
end;

function read_tree(const filename: string): TTree;
var
    t: TTree;
    fp: TextFile;
    line, id: string;
    vals:TStringArray;
    last_parent, group_size, groups, n, parent, i: longint;
begin
    result := default(TTree);
    //fp := fopen(filename, 'r');
    AssignFile(fp,filename);
    reset(fp);
    last_parent := -1;
    group_size := 0;
    groups := 0;
    n := 0;
    while not EOF(fp) do
        begin
            //id := calloc(256, sizeof(char));
            readln(fp,line);
            parent := -1;
            vals:=line.split([' ']);//, '%s %d', id,  and parent);
            id:=vals[0]; TryStrToInt(trim(vals[1]) ,parent);
            setLength(result.parent,n+1);//.reAllocate(n+1);// := realloc(result.parent, (n+1) * sizeof(int));
            result.parent[n] := parent;
            setLength(result.child, n+1);//.reAllocate(n+1);// := realloc(result.child, (n+1) * sizeof(int));
            result.child[n] := -1;
            insert(id, result.name, n);// := realloc(result.name, (n+1) * sizeof(string));
            result.name[n] := id;
            if parent <> last_parent then
                begin
                    inc(groups);
                    setLength(result.group_offset, groups);//.reAllocate(groups);// := realloc(result.group_offset, groups * sizeof(int));
                    result.group_offset[groups-1] := n-group_size;
                    setLength(result.group_size, groups);//.reAllocate(groups);// := realloc(result.group_size, groups * sizeof(int));
                    result.group_size[groups-1] := group_size;
                    group_size := 0;
                    last_parent := parent
                end;
            setLength(result.group, n+1);//.reAllocate(n+1);// := realloc(result.group, (n+1) * sizeof(int));
            result.group[n] := groups;
            if parent >= 0 then
                result.child[parent] := groups;
            inc(n);
            inc(group_size)
        end;
    inc(groups);
    setLength(result.group_offset, groups);//.reAllocate(groups);// := realloc(result.group_offset, groups * sizeof(int));
    result.group_offset[groups-1] := n-group_size;
    setLength(result.group_size, groups);//.reAllocate(groups) ;//:= realloc(result.group_size, groups * sizeof(int));
    result.group_size[groups-1] := group_size;
    result.n := n;
    result.groups := groups;
    setLength(result.leaf, n);// := TIntegers.Create(n);//, sizeof(int));
    for i := 0 to n -1 do
        result.leaf[i] := 1;
    for i := 0 to n -1 do
        if result.parent[i] >= 0 then
            result.leaf[result.parent[i]] := 0;
    CloseFile(fp);
    //result := AllocMem(1* sizeof(TTree));
    //result[0] := t;
    //exit(tree_ptr)
end;


end.

