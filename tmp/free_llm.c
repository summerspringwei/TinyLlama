Result: assembly: 
0000000000000000 <free>:
0: 	movq	%rdi, %rax
3: 	addq	$-16, %rax
7: 	je	0x3d <free+0x3d>
9: 	movslq	(%rax), %rcx
c: 	movq	(%rip), %rdx  # 0x13 <free+0x13>
13: 	leaq	(%rax,%rcx), %rsi
17: 	cmpq	%rsi, %rdx
1a: 	jbe	0x3d <free+0x3d>
1c: 	nopl	(%rax)
20: 	cmpq	$0, 8(%rsi)
25: 	jne	0x3d <free+0x3d>
27: 	movslq	(%rsi), %rsi
2a: 	movslq	%ecx, %rcx
2d: 	addq	%rsi, %rcx
30: 	movl	%ecx, (%rax)
32: 	movq	%rcx, %rsi
35: 	addq	%rax, %rsi
38: 	cmpq	%rsi, %rdx
3b: 	ja	0x20 <free+0x20>
3d: 	movq	$0, -8(%rdi)
45: 	retq 
### c: #define NULL ((void*)0)
typedef unsigned long size_t;  // Customize by platform.
typedef long intptr_t; typedef unsigned long uintptr_t;
typedef long scalar_t__;  // Either arithmetic or pointer type.
/* By default, we understand bool (as a convenience). */
typedef int bool;
#define false 0
#define true 1

/* Forward declarations */
struct gspca_dev {scalar_t__ usb_err; } ;

/* Variables and functions */
char** table_frame ;
int* uvTable ;
size_t underglowGap ;
size_t usedUVspaceSize ;
int /*<<< orphan*/ ** wTableRingBufferStartEndIsoOutputPorts ;

__attribute__((used)) static void free( struct gspca_dev *g )
{
if (!NULL || g->usb_ERR >= 0){   /* UVCStatus error? */
const char flag = const short tempSuspend[uvTABLE[underglowGap]+1]? 'T' : 'o';
for (const short i=flag; i<=usedUVspaceSize && i <= 0xf0ffu; ++i)                    break; /* skip suspend start/end marker in the stream.. it had better be one of USBSPACE / RINGBUFFERSTOPEN */
while (!i > 0xFFFAUL);          while (*tempSuspend!= '\t');         if (!i--) return;}
wTableRingBufferStart EndOfFrame += offsetWrites + lenCmdList & maskLenInUse[maskId];     while(*++lenCmdList == endcmd);               break;
}