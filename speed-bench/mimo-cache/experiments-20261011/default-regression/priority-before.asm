
ds4_metal.o:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000084d9c <_ds4_gpu_stream_expert_cache_priority>:
   84d9c: d10283ff     	sub	sp, sp, #0xa0
   84da0: 6d032beb     	stp	d11, d10, [sp, #0x30]
   84da4: 6d0423e9     	stp	d9, d8, [sp, #0x40]
   84da8: a90567fa     	stp	x26, x25, [sp, #0x50]
   84dac: a9065ff8     	stp	x24, x23, [sp, #0x60]
   84db0: a90757f6     	stp	x22, x21, [sp, #0x70]
   84db4: a9084ff4     	stp	x20, x19, [sp, #0x80]
   84db8: a9097bfd     	stp	x29, x30, [sp, #0x90]
   84dbc: 910243fd     	add	x29, sp, #0x90
   84dc0: 90000008     	adrp	x8, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84dc4: 91000108     	add	x8, x8, #0x0
   84dc8: 5280c009     	mov	w9, #0x600              ; =1536
   84dcc: 9ba92008     	umaddl	x8, w0, w9, x8
   84dd0: b8615915     	ldr	w21, [x8, w1, uxtw #2]
   84dd4: 90000009     	adrp	x9, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84dd8: 91000129     	add	x9, x9, #0x0
   84ddc: f944ed28     	ldr	x8, [x9, #0x9d8]
   84de0: f9400136     	ldr	x22, [x9]
   84de4: f100011f     	cmp	x8, #0x0
   84de8: fa401ac4     	ccmp	x22, #0x0, #0x4, ne
   84dec: 54000061     	b.ne	0x84df8 <_ds4_gpu_stream_expert_cache_priority+0x5c>
   84df0: 1e6302a0     	ucvtf	d0, w21
   84df4: 1400005d     	b	0x84f68 <_ds4_gpu_stream_expert_cache_priority+0x1cc>
   84df8: aa0203f3     	mov	x19, x2
   84dfc: 2a0003e9     	mov	w9, w0
   84e00: 2a0103ea     	mov	w10, w1
   84e04: 528d800b     	mov	w11, #0x6c00            ; =27648
   84e08: 9bab2128     	umaddl	x8, w9, w11, x8
   84e0c: 52800909     	mov	w9, #0x48               ; =72
   84e10: 9ba92154     	umaddl	x20, w10, w9, x8
   84e14: 90000008     	adrp	x8, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84e18: f9400117     	ldr	x23, [x8]
   84e1c: 90000008     	adrp	x8, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84e20: f9400118     	ldr	x24, [x8]
   84e24: 3940f288     	ldrb	w8, [x20, #0x3c]
   84e28: 7100051f     	cmp	w8, #0x1
   84e2c: 540000e1     	b.ne	0x84e48 <_ds4_gpu_stream_expert_cache_priority+0xac>
   84e30: f9401a88     	ldr	x8, [x20, #0x30]
   84e34: eb16011f     	cmp	x8, x22
   84e38: 54000081     	b.ne	0x84e48 <_ds4_gpu_stream_expert_cache_priority+0xac>
   84e3c: b9403a88     	ldr	w8, [x20, #0x38]
   84e40: 6b15011f     	cmp	w8, w21
   84e44: 540007a0     	b.eq	0x84f38 <_ds4_gpu_stream_expert_cache_priority+0x19c>
   84e48: 90000008     	adrp	x8, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84e4c: fd400108     	ldr	d8, [x8]
   84e50: 1e6302a0     	ucvtf	d0, w21
   84e54: 94000000     	bl	0x84e54 <_ds4_gpu_stream_expert_cache_priority+0xb8>
   84e58: 1e680809     	fmul	d9, d0, d8
   84e5c: f9400699     	ldr	x25, [x20, #0x8]
   84e60: b4000639     	cbz	x25, 0x84f24 <_ds4_gpu_stream_expert_cache_priority+0x188>
   84e64: f9400288     	ldr	x8, [x20]
   84e68: cb0802c8     	sub	x8, x22, x8
   84e6c: 9e630108     	ucvtf	d8, x8
   84e70: fd400a8a     	ldr	d10, [x20, #0x10]
   84e74: 1e7a1000     	fmov	d0, #-0.25000000
   84e78: 1e600900     	fmul	d0, d8, d0
   84e7c: 94000000     	bl	0x84e7c <_ds4_gpu_stream_expert_cache_priority+0xe0>
   84e80: 1e6a0800     	fmul	d0, d0, d10
   84e84: 94000000     	bl	0x84e84 <_ds4_gpu_stream_expert_cache_priority+0xe8>
   84e88: 3d800be0     	str	q0, [sp, #0x20]
   84e8c: fd400e8a     	ldr	d10, [x20, #0x18]
   84e90: d2f7f208     	mov	x8, #-0x4070000000000000 ; =-4643211215818981376
   84e94: 9e670100     	fmov	d0, x8
   84e98: 1e600900     	fmul	d0, d8, d0
   84e9c: 94000000     	bl	0x84e9c <_ds4_gpu_stream_expert_cache_priority+0x100>
   84ea0: 1e600940     	fmul	d0, d10, d0
   84ea4: 94000000     	bl	0x84ea4 <_ds4_gpu_stream_expert_cache_priority+0x108>
   84ea8: 3d8007e0     	str	q0, [sp, #0x10]
   84eac: 4ea81d00     	mov.16b	v0, v8
   84eb0: 94000000     	bl	0x84eb0 <_ds4_gpu_stream_expert_cache_priority+0x114>
   84eb4: 3d8003e0     	str	q0, [sp]
   84eb8: fd401280     	ldr	d0, [x20, #0x20]
   84ebc: 94000000     	bl	0x84ebc <_ds4_gpu_stream_expert_cache_priority+0x120>
   84ec0: 9000001a     	adrp	x26, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84ec4: 9100035a     	add	x26, x26, #0x0
   84ec8: ad400be3     	ldp	q3, q2, [sp]
   84ecc: 6e180403     	mov.d	v3[1], v0[0]
   84ed0: ad400344     	ldp	q4, q0, [x26]
   84ed4: 3dc00be1     	ldr	q1, [sp, #0x20]
   84ed8: 6e180441     	mov.d	v1[1], v2[0]
   84edc: 3d800be1     	str	q1, [sp, #0x20]
   84ee0: 6e60dc60     	fmul.2d	v0, v3, v0
   84ee4: ad0003e4     	stp	q4, q0, [sp]
   84ee8: fd401348     	ldr	d8, [x26, #0x20]
   84eec: fd401680     	ldr	d0, [x20, #0x28]
   84ef0: 94000000     	bl	0x84ef0 <_ds4_gpu_stream_expert_cache_priority+0x154>
   84ef4: 6d428b41     	ldp	d1, d2, [x26, #0x28]
   84ef8: f100073f     	cmp	x25, #0x1
   84efc: 6f00e403     	movi.2d	v3, #0000000000000000
   84f00: 1e620c62     	fcsel	d2, d3, d2, eq
   84f04: ad408fe5     	ldp	q5, q3, [sp, #0x10]
   84f08: 3dc003e4     	ldr	q4, [sp]
   84f0c: 4e64cc65     	fmla.2d	v5, v3, v4
   84f10: 7e70d8a3     	faddp.2d	d3, v5
   84f14: 1f480c00     	fmadd	d0, d0, d8, d3
   84f18: 1e612841     	fadd	d1, d2, d1
   84f1c: 1e692821     	fadd	d1, d1, d9
   84f20: 1e612809     	fadd	d9, d0, d1
   84f24: fd002289     	str	d9, [x20, #0x40]
   84f28: f9001a96     	str	x22, [x20, #0x30]
   84f2c: b9003a95     	str	w21, [x20, #0x38]
   84f30: 52800028     	mov	w8, #0x1                ; =1
   84f34: 3900f288     	strb	w8, [x20, #0x3c]
   84f38: cb130308     	sub	x8, x24, x19
   84f3c: b40000b7     	cbz	x23, 0x84f50 <_ds4_gpu_stream_expert_cache_priority+0x1b4>
   84f40: d350fd09     	lsr	x9, x8, #16
   84f44: b5000069     	cbnz	x9, 0x84f50 <_ds4_gpu_stream_expert_cache_priority+0x1b4>
   84f48: fc687ae0     	ldr	d0, [x23, x8, lsl #3]
   84f4c: 14000003     	b	0x84f58 <_ds4_gpu_stream_expert_cache_priority+0x1bc>
   84f50: 9e43e100     	ucvtf	d0, x8, #0x8
   84f54: 94000000     	bl	0x84f54 <_ds4_gpu_stream_expert_cache_priority+0x1b8>
   84f58: 90000008     	adrp	x8, 0x84000 <_ds4_gpu_tensor_live_insert_locked+0x1d8>
   84f5c: fd400101     	ldr	d1, [x8]
   84f60: fd402282     	ldr	d2, [x20, #0x40]
   84f64: 1f400820     	fmadd	d0, d1, d0, d2
   84f68: a9497bfd     	ldp	x29, x30, [sp, #0x90]
   84f6c: a9484ff4     	ldp	x20, x19, [sp, #0x80]
   84f70: a94757f6     	ldp	x22, x21, [sp, #0x70]
   84f74: a9465ff8     	ldp	x24, x23, [sp, #0x60]
   84f78: a94567fa     	ldp	x26, x25, [sp, #0x50]
   84f7c: 6d4423e9     	ldp	d9, d8, [sp, #0x40]
   84f80: 6d432beb     	ldp	d11, d10, [sp, #0x30]
   84f84: 910283ff     	add	sp, sp, #0xa0
   84f88: d65f03c0     	ret
