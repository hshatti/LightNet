unit box;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}


interface
uses SysUtils, Math, utils, lightnet;

type


  PSortableBBox = ^TSortableBBox;
  TSortableBBox = record
      index, class_id : longint;
      probs : PPSingle;
  end;

  TDBox = record
    dx, dy, dw, dh :Single;
  end;

  PDetectionWithClass = ^TDetectionWithClass;
  TDetectionWithClass = record
      det:TDetection;
      // The most probable class id: the best class index in this->prob.
      // Is filled temporary when processing results, otherwise not initialized
      best_class: longint;
  end;


function nms_comparator(const pa, pb:TSortableBBox):longint;
function box_rmse(const a, b: TBox):single;
function diou(const a, b: TBox):TDBox;
function encode_box(const b, anchor: TBox):TBox;
function decode_box(const b, anchor: TBox):TBox;
procedure do_nms_obj(const dets: TArray<TDetection>; total: longint; const classes: longint; const thresh: single);
procedure do_nms_sort(const dets: TArray<TDetection>; total, classes: longint; const thresh: single);
function float_to_box(const f: PSingle):TBox;
function float_to_box_stride(const f: PSingle; const stride: longint):TBox;
function derivative(const a, b: TBox):TDBox;
function box_c(const a, b: TBox):TBoxAbs;
function to_tblr(const a: TBox):TBoxAbs;   function overlap(const x1, w1, x2, w2: single):single;
function box_intersection(const a, b: TBox):single;
function box_union(const a, b: TBox):single;
function box_iou_kind(const a, b: TBox; const iou_kind: TIOULoss):single;
function box_iou(const a, b: TBox):single;
function box_giou(const a, b: TBox):single;
function box_diou(const a, b: TBox):single;
function box_diounms(const a, b: TBox; const beta1: single):single;
function box_ciou(const a, b: TBox):single;
function dintersect(const a, b: TBox):TDBox;
function dunion(const a, b: TBox):TDBox;
function dx_box_iou(const pred, truth: TBox; const iou_loss: TIOULoss):TDxrep;
procedure test_dunion();
procedure test_dintersect();
procedure test_box();
procedure do_nms(const boxes: TArray<TBox>; const probs: TSingles2D; const total, classes: longint; const thresh: single);
procedure diounms_sort(const dets: PDetection; total:longint; const classes: longint; const thresh: single; const nms_kind: TNMSKind; const beta1: single);

implementation


function nms_comparator(const pa, pb:TSortableBBox):longint;
var diff : single;
begin
    //sortable_bbox a = *(sortable_bbox *)pa;
    //sortable_bbox b = *(sortable_bbox *)pb;
    diff := pa.probs[pa.index][pb.class_id] - pb.probs[pb.index][pb.class_id];
    if diff < 0 then exit(1)
    else if diff > 0 then exit(-1);
    exit(0);
end;

//function nms_comparator(const a , b: TDetection): longint;
//var
//    diff: single;
//begin
//    diff := 0;
//    if (b.sort_class >= 0) then
//        diff := a.prob[b.sort_class]-b.prob[b.sort_class]
//    else
//        diff := a.objectness-b.objectness;
//    if diff < 0 then
//        exit(1)
//    else
//        if diff > 0 then
//            exit(-1);
//    exit(0)
//end;

procedure do_nms_sort_v2(const boxes: PBox; const probs: PPsingle; const total, classes: longint; const thresh: single);
var
    i, j, k: longint;
    a, b: TBox;
    s : TArray<TSortableBBox>;
begin
    //sortable_bbox * s := sortable_bbox(xcalloc(total, sizeof(sortable_bbox)));
    setLength(s, total);
    for i := 0 to total -1 do
        begin
            s[i].index := i;
            s[i].class_id := 0;
            s[i].probs := probs
        end;
    for k := 0 to classes -1 do
        begin
            for i := 0 to total -1 do
                s[i].class_id := k;
            TTools<TSortableBBox>.QuickSort(@s[0], 0, total-1, nms_comparator);
            for i := 0 to total -1 do
                begin
                    if probs[s[i].index][k] = 0 then
                        continue;
                    a := boxes[s[i].index];
                    for j := i+1 to total -1 do
                        begin
                            b := boxes[s[j].index];
                            if box_iou(a, b) > thresh then
                                probs[s[j].index][k] := 0
                        end
                end
        end;
    //free(s)
end;

function nms_comparator_v3(const a, b: TDetection):longint;
var
    diff: single;
begin
    diff := 0;
    if b.sort_class >= 0 then
        diff := a.prob[b.sort_class]-b.prob[b.sort_class]
    else
        diff := a.objectness-b.objectness;
    if diff < 0 then
        exit(1)
    else
        if diff > 0 then
            exit(-1);
    exit(0)
end;


procedure do_nms_obj(const dets: TArray<TDetection>; total: longint; const classes: longint; const thresh: single);
var
    i, j, k: longint;
    swap: TDetection;
    a, b: TBox;
begin
    k := total-1;
    i := 0;
    while i<= k do begin
        if dets[i].objectness = 0 then
            begin
                swap := dets[i];
                dets[i] := dets[k];
                dets[k] := swap;
                dec(k);
                dec(i)
            end;
      inc(i)
    end;
    total := k+1;
    for i := 0 to total -1 do
        dets[i].sort_class := -1;

    TTools<TDetection>.QuickSort(@dets[0],0, total-1, nms_comparator_v3);
    for i := 0 to total -1 do
        begin
            if dets[i].objectness = 0 then
                continue;
            a := dets[i].bbox;
            for j := i+1 to total -1 do
                begin
                    if dets[j].objectness = 0 then
                        continue;
                    b := dets[j].bbox;
                    if box_iou(a, b) > thresh then
                        begin
                            dets[j].objectness := 0;
                            for k := 0 to classes -1 do
                                dets[j].prob[k] := 0
                        end
                end
        end
end;

procedure do_nms_sort(const dets: TArray<TDetection>; total, classes: longint;
  const thresh: single);
var
    i: longint;
    j: longint;
    k: longint;
    swap: TDetection;
    a: TBox;
    b: TBox;
begin
    k := total-1;
    i:=0;
    while i <= k do begin
        if dets[i].objectness = 0 then
            begin
                swap := dets[i];
                dets[i] := dets[k];
                dets[k] := swap;
                dec(k);
                dec(i)
            end;
      inc(i)
    end;
    total := k+1;
    for k := 0 to classes -1 do
        begin
            for i := 0 to total -1 do
                dets[i].sort_class := k;
            TTools<TDetection>.QuickSort(@dets[0], 0, total-1, nms_comparator_v3);
            for i := 0 to total -1 do
                begin
                    if dets[i].prob[k] = 0 then
                        continue;
                    a := dets[i].bbox;
                    for j := i+1 to total -1 do
                        begin
                            b := dets[j].bbox;
                            if box_iou(a, b) > thresh then
                                dets[j].prob[k] := 0
                        end
                end
        end
end;

function float_to_box(const f: PSingle): TBox;
begin
    result.x := f[0];
    result.y := f[1];
    result.w := f[2];
    result.h := f[3];
end;

function float_to_box_stride(const f: PSingle; const stride: longint): TBox;
begin
    result:=Default(TBox);
    result.x := f[0];
    result.y := f[1 * stride];
    result.w := f[2 * stride];
    result.h := f[3 * stride];
end;

function derivative(const a, b: TBox): TDBox;
begin
    result.dx := 0;
    result.dw := 0;
    result.dy := 0;
    result.dh := 0;
    if a.x < b.x  then result.dx := 1.0 else result.dx := -1.0;
    if a.y < b.y  then result.dy := 1.0 else result.dy := -1.0;
    if a.w < b.w  then result.dw := 1.0 else result.dw := -1.0;
    if a.h < b.h  then result.dh := 1.0 else result.dh := -1.0;
end;

{
var
    d: TDBox;
    l1: single;
    l2: single;
    r1: single;
    r2: single;
    t1: single;
    t2: single;
    b1: single;
    b2: single;
begin
    d.dx := 0;
    d.dw := 0;
    l1 := a.x-a.w / 2;
    l2 := b.x-b.w / 2;
    if l1 > l2 then
        begin
            d.dx := d.dx - 1;
            d.dw := d.dw + 0.5
        end;
    r1 := a.x+a.w / 2;
    r2 := b.x+b.w / 2;
    if r1 < r2 then
        begin
            d.dx := d.dx + 1;
            d.dw := d.dw + 0.5
        end;
    if l1 > r2 then
        begin
            d.dx := -1;
            d.dw := 0
        end;
    if r1 < l2 then
        begin
            d.dx := 1;
            d.dw := 0
        end;
    d.dy := 0;
    d.dh := 0;
    t1 := a.y-a.h / 2;
    t2 := b.y-b.h / 2;
    if t1 > t2 then
        begin
            d.dy := d.dy - 1;
            d.dh := d.dh + 0.5
        end;
    b1 := a.y+a.h / 2;
    b2 := b.y+b.h / 2;
    if b1 < b2 then
        begin
            d.dy := d.dy + 1;
            d.dh := d.dh + 0.5
        end;
    if t1 > b2 then
        begin
            d.dy := -1;
            d.dh := 0
        end;
    if b1 < t2 then
        begin
            d.dy := 1;
            d.dh := 0
        end;
    exit(d)
end;
}

function box_c(const a, b: TBox):TBoxAbs;
begin
    result.top := min(a.y-a.h / 2, b.y-b.h / 2);
    result.bot := max(a.y+a.h / 2, b.y+b.h / 2);
    result.left := min(a.x-a.w / 2, b.x-b.w / 2);
    result.right := max(a.x+a.w / 2, b.x+b.w / 2);
end;

function to_tblr(const a: TBox):TBoxAbs;
var
    t, b, l, r: single;
begin
    result := default(TBoxAbs);
    t := a.y-(a.h / 2);
    b := a.y+(a.h / 2);
    l := a.x-(a.w / 2);
    r := a.x+(a.w / 2);
    result.top := t;
    result.bot := b;
    result.left := l;
    result.right := r
end;

function overlap(const x1, w1, x2, w2: single): single;
var
    l1: single;
    l2: single;
    left: single;
    r1: single;
    r2: single;
    right: single;
begin
    l1 := x1-w1 / 2;
    l2 := x2-w2 / 2;
    if (l1 > l2) then left := l1 else left := l2;
    r1 := x1+w1 / 2;
    r2 := x2+w2 / 2;
    if r1 < r2 then right := r1 else right := r2;
    exit(right-left)
end;

function box_intersection(const a, b: TBox): single;
var
    w,h,area: single;
begin
    w := overlap(a.x, a.w, b.x, b.w);
    h := overlap(a.y, a.h, b.y, b.h);
    if (w < 0) or (h < 0) then
        exit(0);
    area := w * h;
    exit(area)
end;

function box_union(const a, b: TBox): single;
var
    i: single;
    u: single;
begin
    i := box_intersection(a, b);
    u := a.w * a.h+b.w * b.h-i;
    exit(u)
end;

function box_iou_kind(const a, b: TBox; const iou_kind: TIOULoss):single;
begin
    case iou_kind of
        ilIOU:
            exit(box_iou(a, b));
        ilGIOU:
            exit(box_giou(a, b));
        ilDIOU:
            exit(box_diou(a, b));
        ilCIOU:
            exit(box_ciou(a, b))
    end;
    exit(box_iou(a, b))
end;

function box_iou(const a, b: TBox): single;
var I,U:single;
begin
    //exit(box_intersection(a, b) / box_union(a, b))
    I := box_intersection(a, b);
    U := box_union(a, b);
    if (I = 0) or (U = 0) then
        exit( 0);
    exit( I / U);
end;

function box_giou(const a, b: TBox):single;
var
    ba: TBoxAbs;
    w: single;
    h: single;
    c: single;
    iou: single;
    u: single;
    giou_term: single;
begin
    ba := box_c(a, b);
    w := ba.right-ba.left;
    h := ba.bot-ba.top;
    c := w * h;
    iou := box_iou(a, b);
    if c = 0 then
        exit(iou);
    u := box_union(a, b);
    giou_term := (c-u) / c;
  {$ifdef DEBUG}
    writeln('  c: %f, u: %f, giou_term: %f', c, u, giou_term);
  {$endif}
    exit(iou-giou_term)
end;

function box_diou(const a, b: TBox):single;
var
    ba: TBoxAbs;
    w: single;
    h: single;
    c: single;
    iou: single;
    d: single;
    u: single;
    diou_term: single;
begin
    ba := box_c(a, b);
    w := ba.right-ba.left;
    h := ba.bot-ba.top;
    c := w * w+h * h;
    iou := box_iou(a, b);
    if c = 0 then
        exit(iou);
    d := (a.x-b.x) * (a.x-b.x)+(a.y-b.y) * (a.y-b.y);
    u := power(d / c, 0.6);
    diou_term := u;
    {$ifdef DEBUG}
    writeln('  c: %f, u: %f, riou_term: %f', c, u, diou_term);
    {$endif}
    exit(iou-diou_term)
end;

function box_diounms(const a, b: TBox; const beta1: single):single;
var
    ba: TBoxAbs;
    w: single;
    h: single;
    c: single;
    iou: single;
    d: single;
    u: single;
    diou_term: single;
begin
    ba := box_c(a, b);
    w := ba.right-ba.left;
    h := ba.bot-ba.top;
    c := w * w+h * h;
    iou := box_iou(a, b);
    if c = 0 then
        exit(iou);
    d := (a.x-b.x) * (a.x-b.x)+(a.y-b.y) * (a.y-b.y);
    u := power(d / c, beta1);
    diou_term := u;
    {$ifdef DEBUG}
    writeln('  c: %f, u: %f, riou_term: %f', c, u, diou_term);
    {$endif}
    exit(iou-diou_term)
end;

function box_ciou(const a, b: TBox):single;
var
    ba: TBoxAbs;
    w: single;
    h: single;
    c: single;
    iou: single;
    u: single;
    d: single;
    ar_gt: single;
    ar_pred: single;
    ar_loss: single;
    alpha: single;
    ciou_term: single;
begin
    ba := box_c(a, b);
    w := ba.right-ba.left;
    h := ba.bot-ba.top;
    c := w * w+h * h;
    iou := box_iou(a, b);
    if c = 0 then
        exit(iou);
    u := (a.x-b.x) * (a.x-b.x)+(a.y-b.y) * (a.y-b.y);
    d := u / c;
    ar_gt := b.w / b.h;
    ar_pred := a.w / a.h;
    ar_loss := 4 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * (arctan(ar_gt)-arctan(ar_pred));
    alpha := ar_loss / (1-iou+ar_loss+0.000001);
    ciou_term := d+alpha * ar_loss;
    {$ifdef DEBUG}
    writeln('  c: %f, u: %f, riou_term: %f', c, u, ciou_term);
    {$endif}
    exit(iou-ciou_term)
end;

function dx_box_iou(const pred, truth: TBox; const iou_loss: TIOULoss):TDxrep;
var
    pred_tblr, truth_tblr: TBoxabs;
    pred_t: single;
    pred_b: single;
    pred_l: single;
    pred_r: single;
    X: single;
    Xhat: single;
    Ih: single;
    Iw: single;
    I: single;
    U: single;
    S: single;
    giou_Cw: single;
    giou_Ch: single;
    giou_C: single;
    dX_wrt_t: single;
    dX_wrt_b: single;
    dX_wrt_l: single;
    dX_wrt_r: single;
    dI_wrt_t: single;
    dI_wrt_b: single;
    dI_wrt_l: single;
    dI_wrt_r: single;
    dU_wrt_t: single;
    dU_wrt_b: single;
    dU_wrt_l: single;
    dU_wrt_r: single;
    dC_wrt_t: single;
    dC_wrt_b: single;
    dC_wrt_l: single;
    dC_wrt_r: single;
    p_dt: single;
    p_db: single;
    p_dl: single;
    p_dr: single;
    Ct: single;
    Cb: single;
    Cl: single;
    Cr: single;
    Cw: single;
    Ch: single;
    C: single;
    dCt_dx: single;
    dCt_dy: single;
    dCt_dw: single;
    dCt_dh: single;
    dCb_dx: single;
    dCb_dy: single;
    dCb_dw: single;
    dCb_dh: single;
    dCl_dx: single;
    dCl_dy: single;
    dCl_dw: single;
    dCl_dh: single;
    dCr_dx: single;
    dCr_dy: single;
    dCr_dw: single;
    dCr_dh: single;
    dCw_dx: single;
    dCw_dy: single;
    dCw_dw: single;
    dCw_dh: single;
    dCh_dx: single;
    dCh_dy: single;
    dCh_dw: single;
    dCh_dh: single;
    p_dx: single;
    p_dy: single;
    p_dw: single;
    p_dh: single;
    ar_gt: single;
    ar_pred: single;
    ar_loss: single;
    alpha: single;
    ar_dw: single;
    ar_dh: single;
begin
    pred_tblr := to_tblr(pred);
    pred_t := math.min(pred_tblr.top, pred_tblr.bot);
    pred_b := math.max(pred_tblr.top, pred_tblr.bot);
    pred_l := math.min(pred_tblr.left, pred_tblr.right);
    pred_r := math.max(pred_tblr.left, pred_tblr.right);
    truth_tblr := to_tblr(truth);
    writeln(#10'iou: %f, giou: %f', box_iou(pred, truth), box_giou(pred, truth));
    writeln('pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)', pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    writeln('truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)', truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    result := Default(TDxrep);
    X := (pred_b-pred_t) * (pred_r-pred_l);
    Xhat := (truth_tblr.bot-truth_tblr.top) * (truth_tblr.right-truth_tblr.left);
    Ih := math.min(pred_b, truth_tblr.bot) - math.max(pred_t, truth_tblr.top);
    Iw := math.min(pred_r, truth_tblr.right) - math.max(pred_l, truth_tblr.left);
    I := Iw * Ih;
    U := X+Xhat-I;
    S := (pred.x-truth.x) * (pred.x-truth.x)+(pred.y-truth.y) * (pred.y-truth.y);
    giou_Cw := math.max(pred_r, truth_tblr.right)-math.min(pred_l, truth_tblr.left);
    giou_Ch := math.max(pred_b, truth_tblr.bot)-math.min(pred_t, truth_tblr.top);
    giou_C := giou_Cw * giou_Ch;
    dX_wrt_t := -1 * (pred_r-pred_l);
    dX_wrt_b := pred_r-pred_l;
    dX_wrt_l := -1 * (pred_b-pred_t);
    dX_wrt_r := pred_b-pred_t;
    if (pred_t > truth_tblr.top) then
        dI_wrt_t := (-1 * Iw)
    else
        dI_wrt_t := 0;
    if (pred_b < truth_tblr.bot) then
        dI_wrt_b := Iw
    else
        dI_wrt_b := 0;
    if (pred_l > truth_tblr.left) then
        dI_wrt_l := (-1 * Ih)
    else
        dI_wrt_l := 0;
    if (pred_r < truth_tblr.right) then
        dI_wrt_r := Ih
    else
        dI_wrt_r := 0;
    dU_wrt_t := dX_wrt_t-dI_wrt_t;
    dU_wrt_b := dX_wrt_b-dI_wrt_b;
    dU_wrt_l := dX_wrt_l-dI_wrt_l;
    dU_wrt_r := dX_wrt_r-dI_wrt_r;
    if (pred_t < truth_tblr.top) then
        dC_wrt_t := (-1 * giou_Cw)
    else
        dC_wrt_t := 0;
    if (pred_b > truth_tblr.bot) then
        dC_wrt_b := giou_Cw
    else
        dC_wrt_b := 0;
    if (pred_l < truth_tblr.left) then
        dC_wrt_l := (-1 * giou_Ch)
    else
        dC_wrt_l := 0;
    if (pred_r > truth_tblr.right) then
        dC_wrt_r := giou_Ch
    else
        dC_wrt_r := 0;
    p_dt := 0;
    p_db := 0;
    p_dl := 0;
    p_dr := 0;
    if U > 0 then
        begin
            p_dt := ((U * dI_wrt_t)-(I * dU_wrt_t)) / (U * U);
            p_db := ((U * dI_wrt_b)-(I * dU_wrt_b)) / (U * U);
            p_dl := ((U * dI_wrt_l)-(I * dU_wrt_l)) / (U * U);
            p_dr := ((U * dI_wrt_r)-(I * dU_wrt_r)) / (U * U)
        end;
    if (pred_tblr.top < pred_tblr.bot) then
        p_db := p_db
    else
        p_db := p_dt;
    if (pred_tblr.left < pred_tblr.right) then
        p_dl := p_dl
    else
        p_dl := p_dr;
    if (pred_tblr.left < pred_tblr.right) then
        p_dr := p_dr
    else
        p_dr := p_dl;
    if iou_loss = ilGIOU then
        begin
            if giou_C > 0 then
                begin
                    p_dt := p_dt + (((giou_C * dU_wrt_t)-(U * dC_wrt_t)) / (giou_C * giou_C));
                    p_db := p_db + (((giou_C * dU_wrt_b)-(U * dC_wrt_b)) / (giou_C * giou_C));
                    p_dl := p_dl + (((giou_C * dU_wrt_l)-(U * dC_wrt_l)) / (giou_C * giou_C));
                    p_dr := p_dr + (((giou_C * dU_wrt_r)-(U * dC_wrt_r)) / (giou_C * giou_C))
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dt := ((giou_C * dU_wrt_t)-(U * dC_wrt_t)) / (giou_C * giou_C);
                    p_db := ((giou_C * dU_wrt_b)-(U * dC_wrt_b)) / (giou_C * giou_C);
                    p_dl := ((giou_C * dU_wrt_l)-(U * dC_wrt_l)) / (giou_C * giou_C);
                    p_dr := ((giou_C * dU_wrt_r)-(U * dC_wrt_r)) / (giou_C * giou_C)
                end
        end;
    Ct := min(pred.y-pred.h / 2, truth.y-truth.h / 2);
    Cb := max(pred.y+pred.h / 2, truth.y+truth.h / 2);
    Cl := min(pred.x-pred.w / 2, truth.x-truth.w / 2);
    Cr := max(pred.x+pred.w / 2, truth.x+truth.w / 2);
    Cw := Cr-Cl;
    Ch := Cb-Ct;
    C := Cw * Cw+Ch * Ch;
    dCt_dx := 0;
    if (pred_t < truth_tblr.top) then
        dCt_dy := 1
    else
        dCt_dy := 0;
    dCt_dw := 0;
    if (pred_t < truth_tblr.top) then
        dCt_dh := -0.5
    else
        dCt_dh := 0;
    dCb_dx := 0;
    if (pred_b > truth_tblr.bot) then
        dCb_dy := 1
    else
        dCb_dy := 0;
    dCb_dw := 0;
    if (pred_b > truth_tblr.bot) then
        dCb_dh := 0.5
    else
        dCb_dh := 0;
    if (pred_l < truth_tblr.left) then
        dCl_dx := 1
    else
        dCl_dx := 0;
    dCl_dy := 0;
    if (pred_l < truth_tblr.left) then
        dCl_dw := -0.5
    else
        dCl_dw := 0;
    dCl_dh := 0;
    if (pred_r > truth_tblr.right) then
        dCr_dx := 1
    else
        dCr_dx := 0;
    dCr_dy := 0;
    if (pred_r > truth_tblr.right) then
        dCr_dw := 0.5
    else
        dCr_dw := 0;
    dCr_dh := 0;
    dCw_dx := dCr_dx-dCl_dx;
    dCw_dy := dCr_dy-dCl_dy;
    dCw_dw := dCr_dw-dCl_dw;
    dCw_dh := dCr_dh-dCl_dh;
    dCh_dx := dCb_dx-dCt_dx;
    dCh_dy := dCb_dy-dCt_dy;
    dCh_dw := dCb_dw-dCt_dw;
    dCh_dh := dCb_dh-dCt_dh;
    p_dx := 0;
    p_dy := 0;
    p_dw := 0;
    p_dh := 0;
    p_dx := p_dl+p_dr;
    p_dy := p_dt+p_db;
    p_dw := (p_dr-p_dl);
    p_dh := (p_db-p_dt);
    if iou_loss = ilDIOU then
        begin
            if C > 0 then
                begin
                    p_dx := p_dx + ((2 * (truth.x-pred.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C));
                    p_dy := p_dy + ((2 * (truth.y-pred.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C));
                    p_dw := p_dw + ((2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C));
                    p_dh := p_dh + ((2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C))
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dx := (2 * (truth.x-pred.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C);
                    p_dy := (2 * (truth.y-pred.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C);
                    p_dw := (2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C);
                    p_dh := (2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)
                end
        end;
    if iou_loss = ilCIOU then
        begin
            ar_gt := truth.w / truth.h;
            ar_pred := pred.w / pred.h;
            ar_loss := 4 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * (arctan(ar_gt)-arctan(ar_pred));
            alpha := ar_loss / (1-I / U+ar_loss+0.000001);
            ar_dw := 8 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * pred.h;
            ar_dh := -8 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * pred.w;
            if C > 0 then
                begin
                    p_dx := p_dx + ((2 * (truth.x-pred.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C));
                    p_dy := p_dy + ((2 * (truth.y-pred.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C));
                    p_dw := p_dw + ((2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C)+alpha * ar_dw);
                    p_dh := p_dh + ((2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)+alpha * ar_dh)
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dx := (2 * (truth.x-pred.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C);
                    p_dy := (2 * (truth.y-pred.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C);
                    p_dw := (2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C)+alpha * ar_dw;
                    p_dh := (2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)+alpha * ar_dh
                end
        end;
    result.dt := p_dx;
    result.db := p_dy;
    result.dl := p_dw;
    result.dr := p_dh;
end;


function box_rmse(const a, b: TBox): single;
begin
    exit(
      sqrt( sqr(a.x-b.x{, 2})+
            sqr(a.y-b.y{, 2})+
            sqr(a.w-b.w{, 2})+
            sqr(a.h-b.h{, 2})
            )
      )
end;

function dintersect(const a, b: TBox): TDBox;
var
    w: single;
    h: single;
    dover: TDBox;
begin
    w := overlap(a.x, a.w, b.x, b.w);
    h := overlap(a.y, a.h, b.y, b.h);
    dover := derivative(a, b);
    result.dw := dover.dw * h;
    result.dx := dover.dx * h;
    result.dh := dover.dh * w;
    result.dy := dover.dy * w;
end;

function dunion(const a, b: TBox): TDBox;
var
    di: TDBox;
begin
    di := dintersect(a, b);
    result.dw := a.h-di.dw;
    result.dh := a.w-di.dh;
    result.dx := -di.dx;
    result.dy := -di.dy;
end;

procedure test_dunion();
var
    a: TBox;
    dxa: TBox;
    dya: TBox;
    dwa: TBox;
    dha: TBox;
    b: TBox;
    di: TDBox;
    inter: single;
    xinter: single;
    yinter: single;
    winter: single;
    hinter: single;
begin
      a.x:=0;        a.y:=0;   a.w:=1;   a.h:=1;
    dxa.x:=0.0001; dxa.y:=0; dxa.w:=1; dxa.h:=1;
    dya.x:=0; dya.y:=0.0001; dya.w:=1; dya.h:=1;
    dwa.x:=0; dwa.y:=0; dwa.w:=1.0001; dwa.h:=1;
    dha.x:=0; dha.y:=0; dha.w:=1; dha.h:=1.0001;

    b.x:=0.5; b.y:=0.5; b.w:=0.2; b.h:=0.2;
    di := dunion(a, b);
    writeln(format('Union: %f %f %f %f', [di.dx, di.dy, di.dw, di.dh]));
    inter := box_union(a, b);
    xinter := box_union(dxa, b);
    yinter := box_union(dya, b);
    winter := box_union(dwa, b);
    hinter := box_union(dha, b);
    xinter := (xinter-inter) / (0.0001);
    yinter := (yinter-inter) / (0.0001);
    winter := (winter-inter) / (0.0001);
    hinter := (hinter-inter) / (0.0001);
    writeln(format('Union Manual %f %f %f %f', [xinter, yinter, winter, hinter]))
end;

procedure test_dintersect();
var
    a: TBox;
    dxa: TBox;
    dya: TBox;
    dwa: TBox;
    dha: TBox;
    b: TBox;
    di: TDBox;
    inter: single;
    xinter: single;
    yinter: single;
    winter: single;
    hinter: single;
begin
      a.x:=0;        a.y:=0;   a.w:=1;   a.h:=1;
    dxa.x:=0.0001; dxa.y:=0; dxa.w:=1; dxa.h:=1;
    dya.x:=0; dya.y:=0.0001; dya.w:=1; dya.h:=1;
    dwa.x:=0; dwa.y:=0; dwa.w:=1.0001; dwa.h:=1;
    dha.x:=0; dha.y:=0; dha.w:=1; dha.h:=1.0001;

    b.x:=0.5; b.y:=0.5; b.w:=0.2; b.h:=0.2;
    di := dintersect(a, b);
    writeln(format('Inter: %f %f %f %f', [di.dx, di.dy, di.dw, di.dh]));
    inter := box_intersection(a, b);
    xinter := box_intersection(dxa, b);
    yinter := box_intersection(dya, b);
    winter := box_intersection(dwa, b);
    hinter := box_intersection(dha, b);
    xinter := (xinter-inter) / (0.0001);
    yinter := (yinter-inter) / (0.0001);
    winter := (winter-inter) / (0.0001);
    hinter := (hinter-inter) / (0.0001);
    writeln(format('Inter Manual %f %f %f %f', [xinter, yinter, winter, hinter]))
end;

procedure test_box();
var
    a: TBox;
    dxa: TBox;
    dya: TBox;
    dwa: TBox;
    dha: TBox;
    b: TBox;
    iou: single;
    d: TDBox;
    xiou: single;
    yiou: single;
    wiou: single;
    hiou: single;
begin
    test_dintersect();
    test_dunion();
    a.x:=0;        a.y:=0;   a.w:=1;   a.h:=1;
  dxa.x:=0.00001; dxa.y:=0; dxa.w:=1; dxa.h:=1;
  dya.x:=0; dya.y:=0.00001; dya.w:=1; dya.h:=1;
  dwa.x:=0; dwa.y:=0; dwa.w:=1.00001; dwa.h:=1;
  dha.x:=0; dha.y:=0; dha.w:=1; dha.h:=1.00001;

  b.x:=0.5; b.y:=0.5; b.w:=0.2; b.h:=0.2;
    iou := box_iou(a, b);
    iou := (1-iou) * (1-iou);
    writeln(format('%f', [iou]));
    d := diou(a, b);
    writeln(format('%f %f %f %f', [d.dx, d.dy, d.dw, d.dh]));
    xiou := box_iou(dxa, b);
    yiou := box_iou(dya, b);
    wiou := box_iou(dwa, b);
    hiou := box_iou(dha, b);
    xiou := ((1-xiou) * (1-xiou)-iou) / (0.00001);
    yiou := ((1-yiou) * (1-yiou)-iou) / (0.00001);
    wiou := ((1-wiou) * (1-wiou)-iou) / (0.00001);
    hiou := ((1-hiou) * (1-hiou)-iou) / (0.00001);
    writeln(format('manual %f %f %f %f', [xiou, yiou, wiou, hiou]))
end;

function diou(const a, b: TBox): TDBox;
var
    u: single;
    i: single;
    di: TDBox;
    du: TDBox;
begin
    u := box_union(a, b);
    i := box_intersection(a, b);
    di := dintersect(a, b);
    du := dunion(a, b);
    result := Default(TDBox);
    if (i <= 0) or true then
        begin
            result.dx := b.x-a.x;
            result.dy := b.y-a.y;
            result.dw := b.w-a.w;
            result.dh := b.h-a.h;
            exit
        end;
    result.dx :=  (di.dx * u-du.dx * i) / (u * u);
    result.dy :=  (di.dy * u-du.dy * i) / (u * u);
    result.dw :=  (di.dw * u-du.dw * i) / (u * u);
    result.dh :=  (di.dh * u-du.dh * i) / (u * u);
    // todo [diou] why power of one? there is no need?
    // original
    //dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    //dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    //dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    //dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
end;

procedure do_nms(const boxes: TArray<TBox>; const probs: TSingles2D;
  const total, classes: longint; const thresh: single);
var
    i, j, k: longint;
    any: boolean;
begin
    for i := 0 to total -1 do
        begin
            any := false;
            for k := 0 to classes -1 do
                any := any or (probs[i][k] > 0);
            if not any then
                continue;
            for j := i+1 to total -1 do
                if box_iou(boxes[i], boxes[j]) > thresh then
                    for k := 0 to classes -1 do
                        begin
                            if probs[i][k] < probs[j][k] then
                                probs[i][k] := 0
                            else
                                probs[j][k] := 0
                        end
        end
end;

procedure diounms_sort(const dets: PDetection; total:longint; const classes: longint; const thresh: single; const nms_kind: TNMSKind; const beta1: single);
var
    i, j, k: longint;
    swap: TDetection;
    a, b: TBox;
    sum_prob, alpha_prob, beta_prob: single;
begin
    k := total-1;
    i:=0;
    while i <= k do begin
        if dets[i].objectness = 0 then
            begin
                swap := dets[i];
                dets[i] := dets[k];
                dets[k] := swap;
                dec(k);
                dec(i)
            end;
        inc(i)
    end;
    total := k+1;
    if total=0 then exit;
    for k := 0 to classes -1 do
        begin
            for i := 0 to total -1 do
                dets[i].sort_class := k;
            TTools<TDetection>.QuickSort(@dets[0],0 , total-1, nms_comparator_v3);
            for i := 0 to total -1 do
                begin
                    if dets[i].prob[k] = 0 then
                        continue;
                    a := dets[i].bbox;
                    for j := i+1 to total -1 do
                        begin
                            b := dets[j].bbox;
                            if (box_iou(a, b) > thresh) and (nms_kind = nmsCORNERS_NMS) then
                                begin
                                    sum_prob := sqr(dets[i].prob[k]{, 2})+sqr(dets[j].prob[k]{, 2});
                                    alpha_prob := sqr(dets[i].prob[k]{, 2}) / sum_prob;
                                    beta_prob := sqr(dets[j].prob[k]{, 2}) / sum_prob;
                                    dets[j].prob[k] := 0
                                end
                            else
                                if (box_diou(a, b) > thresh) and (nms_kind = nmsGREEDY_NMS) then
                                    dets[j].prob[k] := 0
                            else
                                if (box_diounms(a, b, beta1) > thresh) and (nms_kind = nmsDIOU_NMS) then
                                    dets[j].prob[k] := 0
                        end
                end
        end
end;

function encode_box(const b, anchor: TBox): TBox;
begin
    result.x := (b.x-anchor.x) / anchor.w;
    result.y := (b.y-anchor.y) / anchor.h;
    result.w := log2(b.w / anchor.w);
    result.h := log2(b.h / anchor.h);
end;

function decode_box(const b, anchor: TBox): TBox;
begin
    result.x := b.x * anchor.w+anchor.x;
    result.y := b.y * anchor.h+anchor.y;
    result.w := power(2.0, b.w) * anchor.w;
    result.h := power(2.0, b.h) * anchor.h;
end;


end.

