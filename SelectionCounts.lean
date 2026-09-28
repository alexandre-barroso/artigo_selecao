import SelectionStability

namespace SelectionCounts
noncomputable section
open SelectionStability

def mass (q T k : ℝ) : ℝ := q / (1+k*T)
def countDen (r1 r2 k1 k2 : ℝ) : ℝ := k2*r2-k1*r1
def levelEstimate (m M d0 r1 r2 k1 k2 : ℝ) : ℝ :=
  clip m M ((r1-r2) / max d0 (countDen r1 r2 k1 k2))

theorem count_den_identity (q T k1 k2 : ℝ)
    (hT : 0<T) (h1 : 0<k1) (h12 : k1<k2) :
    countDen (mass q T k1) (mass q T k2) k1 k2 =
      (k2-k1)*q / ((1+k1*T)*(1+k2*T)) := by
  have h2 : 0<k2 := lt_trans h1 h12
  have hd1 : 1+k1*T ≠ 0 := ne_of_gt (by positivity)
  have hd2 : 1+k2*T ≠ 0 := ne_of_gt (by positivity)
  unfold countDen mass
  field_simp
  ring

theorem count_num_identity (q T k1 k2 : ℝ)
    (hT : 0<T) (h1 : 0<k1) (h12 : k1<k2) :
    mass q T k1-mass q T k2 =
      T*countDen (mass q T k1) (mass q T k2) k1 k2 := by
  have h2 : 0<k2 := lt_trans h1 h12
  have hd1 : 1+k1*T ≠ 0 := ne_of_gt (by positivity)
  have hd2 : 1+k2*T ≠ 0 := ne_of_gt (by positivity)
  unfold countDen mass
  field_simp
  ring

theorem count_den_lower_bound (q qmin T M k1 k2 : ℝ)
    (hq : 0<qmin) (hqq : qmin≤q) (hT : 0<T) (hTM : T≤M)
    (h1 : 0<k1) (h12 : k1<k2) :
    (k2-k1)*qmin/((1+k1*M)*(1+k2*M)) ≤
      countDen (mass q T k1) (mass q T k2) k1 k2 := by
  have h2 : 0<k2 := lt_trans h1 h12
  have hgap : 0<k2-k1 := sub_pos.mpr h12
  have hM : 0<M := lt_of_lt_of_le hT hTM
  have hqp : 0<q := lt_of_lt_of_le hq hqq
  rw [count_den_identity q T k1 k2 hT h1 h12]
  have hnum : (k2-k1)*qmin≤(k2-k1)*q :=
    mul_le_mul_of_nonneg_left hqq (le_of_lt (sub_pos.mpr h12))
  have hd1 : 1+k1*T≤1+k1*M := by nlinarith
  have hd2 : 1+k2*T≤1+k2*M := by nlinarith
  have hden : (1+k1*T)*(1+k2*T)≤(1+k1*M)*(1+k2*M) :=
    mul_le_mul hd1 hd2 (by positivity) (by positivity)
  exact div_le_div₀ (by positivity) hnum (by positivity) hden

theorem lower_projection_distance (d0 x d : ℝ) (hd : d0≤d) :
    |max d0 x-d| ≤ |x-d| := by
  by_cases h : d0≤x
  · rw [max_eq_right h]
  · have hx : x≤d0 := le_of_not_ge h
    rw [max_eq_left hx, abs_of_nonpos (by linarith : d0-d≤0),
        abs_of_nonpos (by linarith : x-d≤0)]
    linarith

theorem projected_inverse_error (m M T v vh d dh d0 ev ed : ℝ)
    (hm : m≤T) (hM : T≤M) (hT : 0≤T) (h0 : 0<d0)
    (hd : d0≤d) (hid : v=T*d)
    (hv : |vh-v|≤ev) (herr : |dh-d|≤ed) :
    |clip m M (vh / max d0 dh)-T| ≤ (ev+M*ed)/d0 := by
  let dt := max d0 dh
  have hdt : d0≤dt := le_max_left _ _
  have hpos : 0<dt := lt_of_lt_of_le h0 hdt
  have hn : dt≠0 := ne_of_gt hpos
  have hproj : |dt-d|≤ed := le_trans (lower_projection_distance d0 dh d hd) herr
  have hed : 0≤ed := le_trans (abs_nonneg _) herr
  have hres : |vh-T*dt|≤ev+M*ed := by
    calc
      |vh-T*dt| = |(vh-v)+(-T)*(dt-d)| := by rw [hid]; congr 1; ring
      _ ≤ |vh-v| + |(-T)*(dt-d)| := abs_add_le _ _
      _ = |vh-v| + T*|dt-d| := by rw [abs_mul, abs_neg, abs_of_nonneg hT]
      _ ≤ ev+M*ed := add_le_add hv (mul_le_mul hM hproj (abs_nonneg _) (le_trans hT hM))
  have hclip := clip_distance m M (vh/dt) T hm hM
  have hidiv : vh/dt-T = (vh-T*dt)/dt := by field_simp
  have hraw : |vh/dt-T|≤(ev+M*ed)/d0 := by
    rw [hidiv, abs_div, abs_of_pos hpos]
    exact div_le_div₀ (le_trans (abs_nonneg _) hres) hres h0 hdt
  exact le_trans hclip hraw

theorem count_den_error (r1 r2 rh1 rh2 k1 k2 e : ℝ)
    (h1 : 0≤k1) (h2 : 0≤k2)
    (he1 : |rh1-r1|≤e) (he2 : |rh2-r2|≤e) :
    |countDen rh1 rh2 k1 k2-countDen r1 r2 k1 k2| ≤ (k1+k2)*e := by
  calc
    _ = |k2*(rh2-r2)+(-k1)*(rh1-r1)| := by unfold countDen; congr 1; ring
    _ ≤ |k2*(rh2-r2)| + |(-k1)*(rh1-r1)| := abs_add_le _ _
    _ = k2*|rh2-r2|+k1*|rh1-r1| := by rw [abs_mul, abs_mul, abs_neg, abs_of_nonneg h1, abs_of_nonneg h2]
    _ ≤ k2*e+k1*e := add_le_add (mul_le_mul_of_nonneg_left he2 h2) (mul_le_mul_of_nonneg_left he1 h1)
    _ = (k1+k2)*e := by ring

theorem retained_level_error (m M T r1 r2 rh1 rh2 k1 k2 d0 e : ℝ)
    (hm : m≤T) (hM : T≤M) (hT : 0≤T)
    (h1 : 0≤k1) (h2 : 0≤k2) (h0 : 0<d0)
    (hd : d0≤countDen r1 r2 k1 k2)
    (hid : r1-r2=T*countDen r1 r2 k1 k2)
    (he1 : |rh1-r1|≤e) (he2 : |rh2-r2|≤e) :
    |levelEstimate m M d0 rh1 rh2 k1 k2-T| ≤
      ((2+M*(k1+k2))/d0)*e := by
  have hv : |(rh1-rh2)-(r1-r2)|≤2*e := by
    calc
      _ = |(rh1-r1)+(-(rh2-r2))| := by congr 1; ring
      _ ≤ |rh1-r1|+|-(rh2-r2)| := abs_add_le _ _
      _ ≤ 2*e := by rw [abs_neg]; linarith
  have hh := projected_inverse_error m M T (r1-r2) (rh1-rh2)
    (countDen r1 r2 k1 k2) (countDen rh1 rh2 k1 k2) d0 (2*e) ((k1+k2)*e)
    hm hM hT h0 hd hid hv (count_den_error r1 r2 rh1 rh2 k1 k2 e h1 h2 he1 he2)
  unfold levelEstimate
  convert hh using 1 <;> ring

theorem retained_level_model_error (m M T q qmin rh1 rh2 k1 k2 e : ℝ)
    (hm : 0<m) (hmT : m≤T) (hTM : T≤M)
    (hq : 0<qmin) (hqq : qmin≤q)
    (h1 : 0<k1) (h12 : k1<k2)
    (he1 : |rh1-mass q T k1|≤e) (he2 : |rh2-mass q T k2|≤e) :
    |levelEstimate m M ((k2-k1)*qmin/((1+k1*M)*(1+k2*M))) rh1 rh2 k1 k2-T| ≤
      ((2+M*(k1+k2))/((k2-k1)*qmin/((1+k1*M)*(1+k2*M))))*e := by
  have hT : 0<T := lt_of_lt_of_le hm hmT
  have hM : 0<M := lt_of_lt_of_le hT hTM
  have h2 : 0<k2 := lt_trans h1 h12
  have hgap : 0<k2-k1 := sub_pos.mpr h12
  exact retained_level_error m M T (mass q T k1) (mass q T k2) rh1 rh2 k1 k2
    ((k2-k1)*qmin/((1+k1*M)*(1+k2*M))) e hmT hTM (le_of_lt hT)
    (le_of_lt h1) (le_of_lt h2) (by positivity)
    (count_den_lower_bound q qmin T M k1 k2 hq hqq hT hTM h1 h12)
    (count_num_identity q T k1 k2 hT h1 h12) he1 he2

theorem geometric_derivative_factor (k m u : ℝ)
    (hk : 0<k) (hm : 0<m) (hu : m≤u) :
    k/(u*(1+k*u)) ≤ 1/m^2 := by
  have hup : 0<u := lt_of_lt_of_le hm hu
  apply (div_le_div_iff₀ (by positivity : 0<u*(1+k*u)) (pow_pos hm 2)).mpr
  have hsq : m^2≤u^2 := by nlinarith
  have hmul := mul_le_mul_of_nonneg_left hsq (le_of_lt hk)
  nlinarith

theorem rate_estimate_error (n S Z : ℝ) (hn : 0<n) (hS : n≤S) (hZ : 0<Z) :
    |n/S-Z| ≤ (Z/n)*|S-n/Z| := by
  have hSp : 0<S := lt_of_lt_of_le hn hS
  have hSn : S≠0 := ne_of_gt hSp
  have hZn : Z≠0 := ne_of_gt hZ
  have hid : n/S-Z = (-Z/S)*(S-n/Z) := by field_simp; ring
  rw [hid, abs_mul, abs_div, abs_neg, abs_of_pos hZ, abs_of_pos hSp]
  exact mul_le_mul_of_nonneg_right
    (div_le_div_of_nonneg_left (le_of_lt hZ) hn hS) (abs_nonneg _)

#print axioms count_den_identity
#print axioms count_num_identity
#print axioms count_den_lower_bound
#print axioms lower_projection_distance
#print axioms projected_inverse_error
#print axioms count_den_error
#print axioms retained_level_error
#print axioms retained_level_model_error
#print axioms geometric_derivative_factor
#print axioms rate_estimate_error
end
end SelectionCounts
