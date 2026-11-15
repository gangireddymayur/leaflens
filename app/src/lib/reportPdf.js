// Builds a one-page (sometimes two) PDF report of a scan. Pure jsPDF, no platform code,
// so it can be tested in Node. Everything is generated on the device.
import { HEALTHY_TIPS, knowledgeFor } from './knowledge.js';

const GREEN = [31, 191, 138];
const DARK = [6, 21, 14];
const TEXT = [30, 41, 59];
const MUTED = [100, 116, 139];
const AMBER = [217, 119, 6];

// jsPDF's built-in fonts cover plain Latin text, so normalise typographic characters.
const clean = (s) => String(s ?? '').replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, '-').replace(/·/g, '-');

export const REPORT_DISCLAIMER =
  'This is a learning project, not agronomic advice. In a test on real-world web photos the top answer was right about 7 times in 10, ' +
  'and the right answer was in the top 3 about 9 times in 10. The model only knows 38 leaf conditions across 14 crops, and photos of ' +
  'anything other than a leaf are unreliable. Confirm with a local agricultural officer before treating a plant.';

export async function buildReportPdf({ photo, result, now = new Date() }) {
  const { jsPDF } = await import('jspdf');
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = 210;
  const H = 297;
  const M = 16;
  const best = result.top[0];
  const unsure = best.probability < 0.6;
  const info = knowledgeFor(best.raw);
  let y = 0;

  // Paint every page white so the text is readable in any PDF viewer, light or dark.
  const paper = () => {
    doc.setFillColor(255, 255, 255);
    doc.rect(0, 0, W, H, 'F');
  };
  paper();

  const footer = () => {
    const pages = doc.getNumberOfPages();
    for (let p = 1; p <= pages; p++) {
      doc.setPage(p);
      doc.setDrawColor(226, 232, 240);
      doc.line(M, H - 16, W - M, H - 16);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(...MUTED);
      doc.text('Generated on this device by LeafLens. Designed and developed by Mayur Gangireddy.', M, H - 11);
      doc.text('(c) 2026 Mayur Gangireddy. All rights reserved.', M, H - 7);
      doc.text(`Page ${p} of ${pages}`, W - M, H - 7, { align: 'right' });
    }
  };
  const space = (needed) => {
    if (y + needed > H - 22) {
      doc.addPage();
      paper();
      y = M;
    }
  };
  const heading = (t) => {
    space(14);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(...GREEN);
    doc.text(clean(t), M, y);
    y += 6;
  };
  const paragraph = (t, { size = 10, color = TEXT, gap = 4 } = {}) => {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(size);
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(clean(t), W - 2 * M);
    for (const line of lines) {
      space(5.2);
      doc.text(line, M, y);
      y += 5;
    }
    y += gap;
  };
  const bullet = (t) => {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(...TEXT);
    const lines = doc.splitTextToSize(clean(t), W - 2 * M - 6);
    lines.forEach((line, i) => {
      space(5.2);
      if (i === 0) doc.text('-', M + 1, y);
      doc.text(line, M + 6, y);
      y += 5;
    });
    y += 1;
  };

  // Header band
  doc.setFillColor(...DARK);
  doc.rect(0, 0, W, 30, 'F');
  doc.setFillColor(...GREEN);
  doc.roundedRect(M, 8, 14, 14, 3.5, 3.5, 'F');
  doc.setTextColor(...DARK);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.text('L', M + 7, 17.6, { align: 'center' });
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(17);
  doc.text('LeafLens', M + 19, 15);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(180, 210, 195);
  doc.text('Leaf health report', M + 19, 21);
  doc.text(now.toLocaleString(), W - M, 15, { align: 'right' });
  y = 40;

  // Photo and result summary
  const PW = 62;
  const PH = Math.min(78, PW * (photo.height / photo.width));
  doc.addImage(photo.dataUrl, 'JPEG', M, y, PW, PH);
  const rx = M + PW + 8;
  const rw = W - M - rx;
  let ry = y + 4;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text(clean(best.crop).toUpperCase(), rx, ry);
  ry += 9;
  doc.setFontSize(20);
  doc.setTextColor(...TEXT);
  const titleLines = doc.splitTextToSize(clean(best.condition), rw);
  doc.text(titleLines, rx, ry);
  ry += titleLines.length * 8 + 3;
  const status = unsure ? 'Not sure about this one' : best.healthy ? 'Looks healthy' : 'Possible issue detected';
  doc.setFontSize(11);
  doc.setTextColor(...(unsure || !best.healthy ? AMBER : GREEN));
  doc.text(status, rx, ry);
  ry += 8;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.setTextColor(...TEXT);
  doc.text(`${Math.round(best.probability * 100)}% confidence`, rx, ry);
  ry += 3;
  doc.setFillColor(226, 232, 240);
  doc.roundedRect(rx, ry, rw, 3, 1.5, 1.5, 'F');
  doc.setFillColor(...(unsure || !best.healthy ? AMBER : GREEN));
  doc.roundedRect(rx, ry, Math.max(2, rw * best.probability), 3, 1.5, 1.5, 'F');
  ry += 12;
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text(`Analysed on-device in ${Math.round(result.ms)} ms. No internet used.`, rx, ry);
  y += PH + 10;

  heading('Other possibilities');
  for (const o of result.top.slice(1)) {
    space(7);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(...TEXT);
    doc.text(`${clean(o.crop)} - ${clean(o.condition)}`, M, y);
    doc.text(`${(o.probability * 100).toFixed(1)}%`, W - M, y, { align: 'right' });
    y += 6;
  }
  y += 4;

  if (unsure) {
    heading('Why no advice is shown');
    paragraph('The model is not confident about this photo. Try a clear, close-up photo of a single leaf in good light.');
  } else if (best.healthy) {
    heading('Keeping it healthy');
    HEALTHY_TIPS.forEach(bullet);
    y += 3;
  } else if (info) {
    heading(`About this condition (${info.kind.toLowerCase()}; ${info.spreadLabel.toLowerCase()})`);
    for (const [label, body] of [['What causes it', info.cause], ['How it spreads', info.spread], ['What to look for', info.signs]]) {
      space(14);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.setTextColor(...TEXT);
      doc.text(label, M, y);
      y += 5;
      paragraph(body, { gap: 2 });
    }
    space(14);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(...TEXT);
    doc.text('What you can do', M, y);
    y += 5;
    info.manage.forEach(bullet);
    y += 3;
  }

  // Limitations box
  const lines = doc.splitTextToSize(clean(REPORT_DISCLAIMER), W - 2 * M - 8);
  const boxH = lines.length * 4.6 + 10;
  space(boxH + 4);
  doc.setFillColor(255, 247, 237);
  doc.setDrawColor(253, 186, 116);
  doc.roundedRect(M, y, W - 2 * M, boxH, 2.5, 2.5, 'FD');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...AMBER);
  doc.text('Please read', M + 4, y + 6);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...TEXT);
  doc.text(lines, M + 4, y + 11);
  y += boxH;

  footer();
  return doc;
}
