// Crime Category Types
export type CrimeCategory =
  | 'Cybercrime'
  | 'Murder'
  | 'Rape'
  | 'Theft'
  | 'Fraud'
  | 'Forgery'
  | 'Bribery'
  | 'Corruption'
  | 'Money laundering'
  | 'Smuggling'
  | 'Terrorism'
  | 'Drug trafficking'
  | 'Acid attack'
  | 'Dowry'
  | 'Domestic violence'
  | 'Human trafficking'
  | 'Hacking'
  | 'Child abuse'
  | 'Food adulteration'
  | 'Land grabbing'
  | 'Pollution'
  | 'Wildlife killing'
  | 'Arson'
  | 'Counterfeiting'
  | 'Illegal construction'
  | 'Fake passport'
  | 'Tax evasion'
  | 'Bank fraud'
  | 'Extortion';

export type CrimeKeywords = Record<CrimeCategory, string[]>;

// Constants
export const CRIME_CATEGORIES: CrimeCategory[] = [
  'Cybercrime',
  'Murder',
  'Rape',
  'Theft',
  'Fraud',
  'Forgery',
  'Bribery',
  'Corruption',
  'Money laundering',
  'Smuggling',
  'Terrorism',
  'Drug trafficking',
  'Acid attack',
  'Dowry',
  'Domestic violence',
  'Human trafficking',
  'Hacking',
  'Child abuse',
  'Food adulteration',
  'Land grabbing',
  'Pollution',
  'Wildlife killing',
  'Arson',
  'Counterfeiting',
  'Illegal construction',
  'Fake passport',
  'Tax evasion',
  'Bank fraud',
  'Extortion',
];

export const CRIME_CATEGORY_LABELS: Record<
  CrimeCategory,
  { label: string; icon: string; color: string }
> = {
  Cybercrime: { label: 'Cybercrime', icon: 'Laptop', color: 'bg-blue-500' },
  Murder: { label: 'Murder', icon: 'Skull', color: 'bg-red-700' },
  Rape: { label: 'Rape', icon: 'AlertTriangle', color: 'bg-red-600' },
  Theft: { label: 'Theft', icon: 'ShoppingBag', color: 'bg-orange-500' },
  Fraud: { label: 'Fraud', icon: 'DollarSign', color: 'bg-yellow-600' },
  Forgery: { label: 'Forgery', icon: 'FileText', color: 'bg-amber-600' },
  Bribery: { label: 'Bribery', icon: 'Banknote', color: 'bg-yellow-700' },
  Corruption: { label: 'Corruption', icon: 'ShieldAlert', color: 'bg-red-800' },
  'Money laundering': { label: 'Money Laundering', icon: 'Coins', color: 'bg-green-700' },
  Smuggling: { label: 'Smuggling', icon: 'Package', color: 'bg-gray-700' },
  Terrorism: { label: 'Terrorism', icon: 'Bomb', color: 'bg-red-900' },
  'Drug trafficking': { label: 'Drug Trafficking', icon: 'Pill', color: 'bg-purple-700' },
  'Acid attack': { label: 'Acid Attack', icon: 'Droplet', color: 'bg-red-600' },
  Dowry: { label: 'Dowry', icon: 'Gift', color: 'bg-pink-600' },
  'Domestic violence': { label: 'Domestic Violence', icon: 'Home', color: 'bg-rose-700' },
  'Human trafficking': { label: 'Human Trafficking', icon: 'Users', color: 'bg-red-800' },
  Hacking: { label: 'Hacking', icon: 'Terminal', color: 'bg-blue-600' },
  'Child abuse': { label: 'Child Abuse', icon: 'Baby', color: 'bg-red-700' },
  'Food adulteration': { label: 'Food Adulteration', icon: 'Utensils', color: 'bg-green-600' },
  'Land grabbing': { label: 'Land Grabbing', icon: 'MapPin', color: 'bg-amber-800' },
  Pollution: { label: 'Pollution', icon: 'Cloud', color: 'bg-gray-600' },
  'Wildlife killing': { label: 'Wildlife Killing', icon: 'Bird', color: 'bg-green-800' },
  Arson: { label: 'Arson', icon: 'Flame', color: 'bg-orange-700' },
  Counterfeiting: { label: 'Counterfeiting', icon: 'Printer', color: 'bg-indigo-600' },
  'Illegal construction': { label: 'Illegal Construction', icon: 'Building', color: 'bg-stone-600' },
  'Fake passport': { label: 'Fake Passport', icon: 'BookOpen', color: 'bg-indigo-700' },
  'Tax evasion': { label: 'Tax Evasion', icon: 'Calculator', color: 'bg-slate-700' },
  'Bank fraud': { label: 'Bank Fraud', icon: 'Landmark', color: 'bg-emerald-700' },
  Extortion: { label: 'Extortion', icon: 'Hand', color: 'bg-orange-800' },
};

// Helper Functions
export function getCrimeIcon(category: CrimeCategory): string {
  return CRIME_CATEGORY_LABELS[category]?.icon || 'AlertCircle';
}

export function getCrimeColor(category: CrimeCategory): string {
  return CRIME_CATEGORY_LABELS[category]?.color || 'bg-gray-500';
}
