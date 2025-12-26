classdef KWaveSource
  %KWaveSource Source wrapper built from a KWaveLinearArray
  % Example:
  %   arr = KWaveLinearArray();
  %   src = KWaveSource(arr, Nt);
  %   rows = src.rows_per_elem{10};
  %   src.p(rows,:) = repmat(myWaveform, numel(rows), 1);

  properties
    p_mask          % Nx-by-Ny logical mask
    p               % [num_active_pixels x Nt] pressure signal matrix
    rows_per_elem   % 1 x num_elems cell, mapping elements to row indices
    src_lin_idx     % linear indices of active pixels
  end

  methods
    function obj = KWaveSource(arrayObj)
      % arrayObj : KWaveLinearArray instance
      % Nt       : number of time samples

      Nx = arrayObj.Nx;
      Ny = arrayObj.Ny;
      num_elems = arrayObj.num_elems;
      elem_centers = arrayObj.elem_centers_px;
      elem_w_px = arrayObj.elem_w_px;
      x_src = arrayObj.x_src;

      % ---- Build p_mask ----
      obj.p_mask = false(Nx, Ny);
      for e = 1:num_elems
        ys = max(1, elem_centers(e) - floor(elem_w_px/2));
        ye = min(Ny, ys + elem_w_px - 1);
        obj.p_mask(x_src, ys:ye) = true;
      end

      % active pixel indices
      obj.src_lin_idx = find(obj.p_mask);

      % ---- rows_per_elem ----
      obj.rows_per_elem = cell(1, num_elems);
      for e = 1:num_elems
        ys = max(1, elem_centers(e) - floor(elem_w_px/2));
        ye = min(Ny, ys + elem_w_px - 1);
        lidx = sub2ind([Nx, Ny], repmat(x_src, 1, ye - ys + 1), ys:ye);
        obj.rows_per_elem{e} = find(ismember(obj.src_lin_idx, lidx));
      end

      % ---- Preallocate p ----
      num_rows = numel(obj.src_lin_idx);
        % pad = arrayObj.Nt - size(obj.p,2);
        % if pad > 0, src.p = [obj.p, zeros(size(obj.p,1), pad, 'like', obj.p)]; end
        % obj.p = src.p;
      obj.p = zeros(num_rows, arrayObj.Nt, arrayObj.data_type);
    end
    function sourceStruct = getSourceStruct(obj)
        sourceStruct.p_mask = obj.p_mask;
        sourceStruct.p = obj.p;
    end
  end
end
